/*
 * Copyright 2015 Gustaf Räntilä
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LIBQ_TEST_FIXTURE_HPP
#define LIBQ_TEST_FIXTURE_HPP

#include <q-test/spy.hpp>

#include <q/promise.hpp>
#include <q/execution_context.hpp>
#include <q/threadpool.hpp>
#include <q/scope.hpp>

#define Q_TEST_MAKE_SCOPE( name ) \
	class name : public ::q::test::fixture { }

namespace q { namespace test {

class fixture
#ifdef QTEST_BACKEND_FIXTURE_CLASS
: public QTEST_BACKEND_FIXTURE_CLASS
#endif
{
public:
	fixture( )
	: mutex_( "q test fixture" )
	, started_( false )
	{
#ifndef QTEST_BACKEND_FIXTURE_SETUP
		qtest_setup( );
#endif
	}

	virtual ~fixture( )
	{
#ifndef QTEST_BACKEND_FIXTURE_TEARDOWN
		qtest_teardown( );
#endif
	}

	Q_MAKE_SIMPLE_EXCEPTION( Error );

	void await_promise( ::q::promise< >&& promise )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		awaiting_promises_.push_back( std::move( promise ) );
	}

QTEST_BACKEND_FIXTURE_SETUP_TEARDOWN_ACCESS:
#ifdef QTEST_BACKEND_FIXTURE_SETUP
	void QTEST_BACKEND_FIXTURE_SETUP( ) override
	{
		qtest_setup( );
	}
#endif

#ifdef QTEST_BACKEND_FIXTURE_TEARDOWN
	void QTEST_BACKEND_FIXTURE_TEARDOWN( ) override
	{
		qtest_teardown( );
	}
#endif

protected:
	virtual void on_setup( ) { }
	virtual void on_teardown( ) { }


	void qtest_setup( )
	{
#ifdef QTEST_BACKEND_REQUIRES_SINGLETON
		qtest_current_test.set_reporter(
			[ this ]( std::exception_ptr e )
			{
				backend_exception_ = std::move( e );
			}
		);
#endif // QTEST_BACKEND_REQUIRES_SINGLETON

		std::tie( bd, queue ) = q::make_event_dispatcher_and_queue<
			q::blocking_dispatcher, q::direct_scheduler
		>( "all" );

		std::tie( tp, tp_queue ) = q::make_event_dispatcher_and_queue<
			q::threadpool, q::direct_scheduler
		>( "test pool", queue, 2 );

		on_setup( );

		if ( !started_ )
			run( q::with( queue ) );
	}

	void qtest_teardown( )
	{
		bool need_to_await_promises;
		{
			Q_AUTO_UNIQUE_LOCK( mutex_ );
			need_to_await_promises = !awaiting_promises_.empty( );
		}

		if ( need_to_await_promises )
			_run( q::with( queue ) );

		on_teardown( );

		tp->terminate( q::termination::linger )
		.finally( [ this ]( )
		{
			tp->await_termination( );
			tp.reset( );
			tp_queue.reset( );
			bd->terminate( q::termination::linger );
		}, queue );

		bd->start( );

		bd->await_termination( );
		bd.reset( );
		queue.reset( );
		test_scopes_.clear( );

#ifdef QTEST_BACKEND_REQUIRES_SINGLETON
		qtest_current_test.unset_reporter( );
		if ( backend_exception_ )
			std::rethrow_exception( backend_exception_ );
#endif // QTEST_BACKEND_REQUIRES_SINGLETON
	}

	void keep_alive( q::scope&& scope )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		test_scopes_.push_back( std::move( scope ) );
	}

	template< typename Promise >
	void _run( Promise&& promise )
	{
		promise
		.strip( )
		.then( [ this ]( )
		{
			// Await all other promises which have been added for
			// awaiting.
			return await_all( );
		} )
		.fail( [ ]( std::exception_ptr e )
		{
			// If a test throws an asynchronous exception, the test
			// fails.
			QTEST_BACKEND_FAIL(
				"Test threw unhandled asychronous exception:"
				<< LIBQ_EOL
				<< q::stream_exception( e )
			);
		} )
		.fail( [ ]( std::exception_ptr e )
		{
			std::cerr
				<< "Unit test backend didn't handle "
				<< "asynchronous exception:"
				<< LIBQ_EOL
				<< q::stream_exception( e );
			std::abort( );
		} )
		.finally( [ this ]( )
		{
			bd->terminate( q::termination::linger );
		} );

		started_ = true;

		bd->start( );
	}

	// We only allow to run promises by r-value reference (i.e. we take
	// ownership).
	template< typename Promise >
	typename std::enable_if<
		q::is_promise< typename std::decay< Promise >::type >::value
		and
		!std::decay< Promise >::type::shared_type::value
		and
		!std::is_lvalue_reference<
			typename std::decay< Promise >::type
		>::value
	>::type
	run( Promise&& promise )
	{
		_run( std::move( promise ) );
	}

	// Shared promises, we can take as r-value references or copies.
	template< typename Promise >
	typename std::enable_if<
		q::is_promise< typename std::decay< Promise >::type >::value
		and
		std::decay< Promise >::type::shared_type::value
	>::type
	run( Promise&& promise )
	{
		_run( std::forward< Promise >( promise ) );
	}

	std::shared_ptr< q::blocking_dispatcher > bd;
	std::shared_ptr< q::threadpool > tp;

	q::queue_ptr queue;
	q::queue_ptr tp_queue;

	q::test::spy spy;

private:
	q::promise< > await_all( )
	{
		std::vector< q::promise< > > promises_;
		{
			Q_AUTO_UNIQUE_LOCK( mutex_ );
			std::swap( promises_, awaiting_promises_ );
		}

		if ( promises_.empty( ) )
			return q::with( queue );

		return q::all( promises_, queue )
		.then( [ this ]( )
		{
			// Recurse and try to await more promises which might
			// have been added since last time.
			return await_all( );
		} );
	}

	q::mutex mutex_;
	std::vector< q::promise< > > awaiting_promises_;

	std::vector< q::scope > test_scopes_;

	std::atomic< bool > started_;

	std::exception_ptr backend_exception_;
};

} } // namespace test, namespace q

#endif // LIBQ_TEST_FIXTURE_HPP
