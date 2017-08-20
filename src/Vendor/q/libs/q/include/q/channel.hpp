/*
 * Copyright 2013 Gustaf Räntilä
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

#ifndef LIBQ_CHANNEL_HPP
#define LIBQ_CHANNEL_HPP

#include <q/exception.hpp>
#include <q/mutex.hpp>
#include <q/promise.hpp>
#include <q/scope.hpp>
#include <q/concurrency.hpp>
#include <q/concurrency_counter.hpp>

#include <list>
#include <queue>
#include <atomic>

namespace q {

Q_MAKE_SIMPLE_EXCEPTION( channel_closed_exception );

template< typename... T >
class readable;

template< typename... T >
class writable;

template< typename... T >
class channel;

namespace detail {

static constexpr std::size_t default_resume_count( std::size_t count )
{
	return count < 3 ? count : ( ( count * 3 ) / 4 );
}

template< typename... T >
struct channel_traits
{
	typedef q::promise< T... > promise_type;
	typedef q::promise< T... > unique_promise_type;
	typedef std::tuple< > promise_tuple_type;
	typedef q::arguments< > promise_arguments_type;
	typedef std::false_type is_shared;

	typedef std::tuple< T... > outer_tuple_type;
	typedef channel_traits< T... > type;
	typedef std::tuple< T... > tuple_type;
	typedef std::false_type is_promise;

	using inner_arguments_type = arguments< T... >;
	typedef tuple_type inner_tuple_type;
};

template< typename... T >
struct channel_traits< q::promise< T... > >
{
	typedef q::promise< T... > promise_type;
	typedef q::promise< T... > unique_promise_type;
	typedef std::tuple< T... > promise_tuple_type;
	typedef arguments< T... > promise_arguments_type;
	typedef std::false_type is_shared;

	typedef std::tuple< promise_type > outer_tuple_type;
	typedef channel_traits< promise_type > type;
	typedef std::tuple< promise_type > tuple_type;
	typedef std::true_type is_promise;

	using inner_arguments_type = arguments< T... >;
	typedef promise_tuple_type inner_tuple_type;
};

template< typename... T >
struct channel_traits< q::shared_promise< T... > >
{
	typedef q::shared_promise< T... > promise_type;
	typedef q::promise< T... > unique_promise_type;
	typedef std::tuple< T... > promise_tuple_type;
	typedef arguments< T... > promise_arguments_type;
	typedef std::true_type is_shared;

	typedef std::tuple< promise_type > outer_tuple_type;
	typedef channel_traits< promise_type > type;
	typedef std::tuple< promise_type > tuple_type;
	typedef std::true_type is_promise;

	using inner_arguments_type = arguments< T... >;
	typedef promise_tuple_type inner_tuple_type;
};

template< typename From, typename To >
struct pipe_helper;

template< typename... From, typename... To >
struct pipe_helper< arguments< From... >, arguments< To... > >
{
	typedef channel_traits< From... > traits;

	typedef is_argument_same_or_convertible_t<
		arguments< From... >,
		arguments< To... >
	> valid;

	static void pipe( readable< From... > r, writable< To... > w )
	{
		typedef typename traits::inner_tuple_type tuple_type;

		typedef q::custom_function< void( ), true, 22 > write_type;

		auto try_write = std::make_shared< write_type >( );

		auto close = [ try_write, w ]( ) mutable
		{
			w.unset_resume_notification( );
			w.close( );
			( *try_write ) = write_type( );
		};

		auto abort = [ try_write, w ]( std::exception_ptr err )
		mutable
		{
			w.unset_resume_notification( );
			w.close( err );
			( *try_write ) = write_type( );
		};

		// This must not run on multiple threads in parallel
		*try_write = [ try_write, r, w, close, abort ]( )
		mutable
		{
			if ( w.should_write( ) )
			{
				auto on_data =
				[ try_write, w ]( tuple_type&& data )
				mutable
				{
					// Write through
					ignore_result(
						w.write(
							std::move( data ) ) );

					// Recurse
					( *try_write )( );
				};

				r.read( std::move( on_data ), close )
				.strip( )
				.fail( [ abort ]( std::exception_ptr err )
				mutable
				{
					abort( err );
				} );
			}
			else if ( w.is_closed( ) )
			{
				auto err = w.get_exception( );

				if ( err )
					r.close( err );
				else
					r.close( );

				close( );
			}
			else
			{
				auto resume = [ try_write, w ]( ) mutable
				{
					w.unset_resume_notification( );
					( *try_write )( );
				};

				w.set_resume_notification( resume, true );
			}
		};

		w.unset_resume_notification( );
		( *try_write )( );
	};
};

template< typename... T >
class shared_channel
: public std::enable_shared_from_this< shared_channel< T... > >
{
public:
	typedef typename detail::channel_traits< T... >::type traits;

	using inner_arguments_type = typename traits::inner_arguments_type;

	typedef std::tuple< T... >     tuple_type;
	typedef detail::defer< T... >  defer_type;
	typedef arguments< T... >      arguments_type;
	typedef shared_channel< T... > self_type;

	// TODO: Combine the two derived types in one union, with enough space
	//       for both and a pointer to the base type (similar to
	//       q::function) and potentially store them in a vector to reduce
	//       heap allocations.
	struct waiter_type
	{
		virtual ~waiter_type( ) { }

		virtual void set_closed( ) = 0;
		virtual void set_exception( std::exception_ptr ) = 0;
		virtual void set_value( tuple_type&& ) = 0;
	};

	struct defer_waiter_type
	: waiter_type
	{
		void set_closed( ) override
		{
			deferred->set_exception(
				std::make_exception_ptr(
					channel_closed_exception( ) ) );
		}
		void set_exception( std::exception_ptr e ) override
		{
			deferred->set_exception( std::move( e ) );
		}
		void set_value( tuple_type&& t ) override
		{
			deferred->set_value( std::move( t ) );
		}

		defer_waiter_type( std::shared_ptr< defer_type > deferred )
		: deferred( deferred )
		{ }

		std::shared_ptr< defer_type > deferred;
	};

	template< typename FnValue, typename FnClosed >
	struct fast_waiter_type_traits
	{
		typedef bool_type<
			arguments_type
			::template is_convertible_to_incl_void<
				arguments_of_t< FnValue >
			>::value
		> assignable_to_value_directly;

		typedef tuple_arguments_of_are_convertible_from_incl_void_t<
			FnValue,
			arguments_type
		> assignable_to_value_by_tuple;

		typedef bool_type<
			inner_arguments_type
			::template is_convertible_to_incl_void<
				arguments_of_t< FnValue >
			>::value
		> inner_assignable_to_value_directly;

		typedef tuple_arguments_of_are_convertible_from_incl_void_t<
			FnValue,
			inner_arguments_type
		> inner_assignable_to_value_by_tuple;

		typedef result_of_is_voidish_or_eventually_voidish_t< FnValue >
			value_has_valid_return;

		typedef typename arguments_of_t< FnClosed >::empty_or_voidish
			closed_has_valid_arguments;

		typedef result_of_is_voidish_or_eventually_voidish_t< FnClosed >
			closed_has_valid_return;

		typedef bool_type<
			(
				assignable_to_value_directly::value
				or
				assignable_to_value_by_tuple::value
			)
			and
			value_has_valid_return::value
			and
			closed_has_valid_arguments::value
			and
			closed_has_valid_return::value
		> callbacks_are_valid;

		typedef bool_type<
			(
				inner_assignable_to_value_directly::value
				or
				inner_assignable_to_value_by_tuple::value
			)
			and
			value_has_valid_return::value
			and
			closed_has_valid_arguments::value
			and
			closed_has_valid_return::value
		> inner_callbacks_are_valid;
	};

	template< typename FnValue, typename FnClosed >
	struct fast_waiter_type
	: waiter_type
	, fast_waiter_type_traits< FnValue, FnClosed >
	{
		typedef detail::defer< bool > result_defer_type;

		void set_closed( ) override
		{
			auto deferred = this->deferred;
			auto fn_closed = std::move( this->fn_closed );

			Q_MOVE_INTO_MOVABLE( fn_closed );

			auto fn = [ deferred, Q_MOVABLE_MOVE( fn_closed ) ]( )
			mutable
			{
				deferred->satisfy(
					promisify(
						deferred->get_queue( ),
						Q_MOVABLE_CONSUME( fn_closed )
					)
					( )
					.then( [ ]( ) { return false; } )
				);
			};

			deferred->get_queue( )->push( std::move( fn ) );
		}
		void set_exception( std::exception_ptr e ) override
		{
			deferred->set_exception( e );
		}
		void set_value( tuple_type&& t ) override
		{
			auto ch = shared_channel.lock( );

			// If the channel still exists (which should always be
			// the case), we'll make a proxy promise which detects
			// exceptions to automatically close the channel.
			q::function< void( std::exception_ptr ) > cleaner;
			if ( ch )
				cleaner = [ ch ]( std::exception_ptr e )
				{
					ch->close( e );
					ch->clear( );
				};
			else
				cleaner = [ ]( std::exception_ptr ) { };

			const queue_ptr& queue = ch
				? ch->default_queue_
				: deferred->get_queue( );

			auto proxy = ::q::make_shared<
				detail::defer< >
			>( queue );

			auto fn_value = std::move( this->fn_value );
			Q_MOVE_INTO_MOVABLE( fn_value );
			Q_MOVE_INTO_MOVABLE( t );

			auto fn =
			[
				proxy,
				Q_MOVABLE_MOVE( fn_value ),
				Q_MOVABLE_MOVE( t )
			]
			( ) mutable
			{
				proxy->set_by_fun(
					Q_MOVABLE_CONSUME( fn_value ),
					Q_MOVABLE_CONSUME( t )
				);
			};

			deferred->get_queue( )->push( std::move( fn ) );

			deferred->satisfy(
				proxy->get_promise( )
				.then( [ ]( ) { return true; } )
				.tap_error( cleaner )
			);
		}

		template< typename _FnValue, typename _FnClosed >
		fast_waiter_type(
			_FnValue&& fn_value,
			_FnClosed&& fn_closed,
			std::shared_ptr< result_defer_type > deferred,
			std::weak_ptr< self_type > shared_channel
		)
		: fn_value( std::forward< _FnValue >( fn_value ) )
		, fn_closed( std::forward< _FnClosed >( fn_closed ) )
		, deferred( deferred )
		, shared_channel( shared_channel )
		{ }

		FnValue fn_value;
		FnClosed fn_closed;
		std::shared_ptr< result_defer_type > deferred;
		std::weak_ptr< self_type > shared_channel;
	};

	shared_channel(
		const queue_ptr& queue,
		std::size_t buffer_count,
		std::size_t resume_count
	)
	: default_queue_( queue )
	, mutex_( Q_HERE, "channel" )
	, close_exception_( std::make_tuple( false, std::exception_ptr( ) ) )
	, closed_( false )
	, paused_( false )
	, buffer_count_( buffer_count )
	, resume_count_( std::min( resume_count, buffer_count ) )
	{ }

	Q_NODISCARD
	std::size_t buffer_count( ) const
	{
		return buffer_count_;
	}

	Q_NODISCARD
	bool is_closed( ) const
	{
		return closed_;
	}

	Q_NODISCARD
	bool has_exception( ) const
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		return std::get< 0 >( close_exception_ );
	}

	Q_NODISCARD
	std::exception_ptr get_exception( ) const
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		return std::get< 1 >( close_exception_ );
	}

	void close( )
	{
		_close( std::make_tuple( false, std::exception_ptr( ) ) );
	}

	template< typename E >
	typename std::enable_if<
		!std::is_same<
			typename std::decay< E >::type,
			std::exception_ptr
		>::value
		and
		!std::is_same<
			typename std::decay< E >::type,
			channel_closed_exception
		>::value
	>::type
	close( E&& e )
	{
		close( std::make_exception_ptr( std::forward< E >( e ) ) );
	}

	template< typename E >
	typename std::enable_if<
		std::is_same<
			typename std::decay< E >::type,
			channel_closed_exception
		>::value
	>::type
	close( E&& e )
	{
		_close( std::make_tuple( false, std::exception_ptr( ) ) );
	}

	template< typename E >
	typename std::enable_if<
		std::is_same<
			typename std::decay< E >::type,
			std::exception_ptr
		>::value
	>::type
	close( E&& e )
	{
		_close( std::make_tuple( true, std::forward< E >( e ) ) );
	}

	Q_NODISCARD
	bool write( tuple_type&& t )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		if ( closed_.load( std::memory_order_seq_cst ) )
			return false;

		if ( waiters_.empty( ) )
		{
			if ( queue_.size( ) >= buffer_count_ )
				paused_ = true;

			queue_.push( std::move( t ) );
		}
		else
		{
			auto waiter = std::move( waiters_.front( ) );
			waiters_.pop_front( );

			waiter->set_value( std::move( t ) );
		}

		return true;
	}

	Q_NODISCARD
	bool write( const tuple_type& t )
	{
		return write( tuple_type( t ) );
	}

	Q_NODISCARD
	promise< T... > read( )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		if ( queue_.empty( ) )
		{
			if ( closed_.load( std::memory_order_seq_cst ) )
				return reject< T... >(
					default_queue_,
					std::get< 0 >( close_exception_ )
					? std::get< 1 >( close_exception_ )
					: std::make_exception_ptr(
						channel_closed_exception( ) )
				);

			auto defer = ::q::make_shared< defer_type >(
				default_queue_ );

			waiters_.push_back(
				::q::make_unique< defer_waiter_type >(
					defer ) );

			return defer->get_promise( );
		}
		else
		{
			tuple_type t = std::move( queue_.front( ) );
			queue_.pop( );

			if ( queue_.size( ) < resume_count_ && paused_ )
			{
				auto self = this->shared_from_this( );
				default_queue_->push( [ self ]( )
				{
					self->resume( );
				} );
			}

			auto defer = ::q::make_shared< defer_type >(
				default_queue_ );

			defer->set_value( std::move( t ) );

			return defer->get_promise( );
		}
	}

	/**
	 * Fast read version, which doesn't use exceptions for close events.
	 *
	 * fn_value  will be called when the next value is available, but not
	 *           if there is no more values and the channel is closed.
	 * fn_closed will be called if the channel is/gets closed (and doesn't
	 *           get a next value), and doesn't contain an error.
	 *
	 * If the channel has/gets an exception, or any of the fn_value or
	 * fn_closed functions throw exceptions, this means the channel is
	 * closed (but fn_closed will not be called, since the channel was
	 * closed with an error), the returned promise will contain this
	 * exception.
	 * Otherwise, the returned promise will resolve after the fn_value or
	 * fn_closed function was called (and potentially awaited it it/they
	 * returned an empty promies.
	 *
	 * NOTE: If the fn_value throws an exception (synchronously or
	 * asynchronously), not only will this exception be propagated to the
	 * returned promise, the channel will be closed with this exception!
	 */
	template< typename FnValue, typename FnClosed >
	Q_NODISCARD
	typename std::enable_if<
		fast_waiter_type_traits<
			decayed_function_t< FnValue >,
			decayed_function_t< FnClosed >
		>::callbacks_are_valid::value,
		promise< bool >
	>::type
	read( FnValue&& fn_value, FnClosed&& fn_closed )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		typedef fast_waiter_type<
			decayed_function_t< FnValue >,
			decayed_function_t< FnClosed >
		> specific_waiter_type;
		typedef typename specific_waiter_type::result_defer_type
			specific_defer_type;

		if ( queue_.empty( ) )
		{
			if ( closed_.load( std::memory_order_seq_cst ) )
			{
				if ( std::get< 0 >( close_exception_ ) )
					// There was a real error
					return reject< bool >(
						default_queue_,
						std::get< 1 >(
							close_exception_ ) );
				else
				{
					// Nicely closed
					auto self = this->shared_from_this( );

					return q::make_promise(
						default_queue_, fn_closed )
					.then( [ ]( ) { return false; } );
				}
			}

			auto defer = ::q::make_shared< specific_defer_type >(
				default_queue_ );

			waiters_.push_back(
				::q::make_unique< specific_waiter_type >(
					std::forward< FnValue >( fn_value ),
					std::forward< FnClosed >( fn_closed ),
					defer,
					this->shared_from_this( )
				)
			);
			resume( );

			return defer->get_promise( );
		}
		else
		{
			tuple_type t = std::move( queue_.front( ) );
			queue_.pop( );

			if ( queue_.size( ) < resume_count_ )
			{
				auto self = this->shared_from_this( );
				default_queue_->push( [ self ]( )
				{
					self->resume( );
				} );
			}

			auto defer = ::q::make_shared< specific_defer_type >(
				default_queue_ );

			specific_waiter_type waiter(
				std::forward< FnValue >( fn_value ),
				std::forward< FnClosed >( fn_closed ),
				defer,
				this->shared_from_this( )
			);

			waiter.set_value( std::move( t ) );

			return defer->get_promise( );
		}
	}

	Q_NODISCARD
	inline bool should_write( ) const
	{
		return !paused_ && !closed_;
	}

	void set_resume_notification( shared_task fn, bool trigger_now )
	{
		shared_task notification;

		{
			Q_AUTO_UNIQUE_LOCK( mutex_ );

			resume_notification_ = fn;

			if ( trigger_now && !should_write( ) )
				notification = resume_notification_;
		}

		if ( notification )
			default_queue_->push( std::move( notification ) );
	}

	void unset_resume_notification( )
	{
		set_resume_notification( shared_task( ), false );
	}

	/**
	 * Triggers (calls) the resume notification callback, if the channel
	 * allows more data to be sent, but not if the channel is "full".
	 */
	void trigger_resume_notification( )
	{
		task notification;

		{
			Q_AUTO_UNIQUE_LOCK( mutex_ );

			if ( !should_write( ) )
				return;

			notification = resume_notification_;
		}

		if ( notification )
			notification( );
	}

	/**
	 * Adds a scope to this channel. This will cause the channel to "own"
	 * the scope, and thereby destruct it when the channel is destructed.
	 */
	void add_scope_until_closed( scope&& scope )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		if ( closed_ )
			// Already closed - don't keep scope
			return;

		scopes_.emplace_back( std::move( scope ) );
	}

	Q_NODISCARD
	const queue_ptr& get_queue( ) const
	{
		return default_queue_;
	}

	// This will be called if this is a promise-valued channel with
	// stop-on-async-errors enabled.
	void clear( )
	{
		Q_AUTO_UNIQUE_LOCK( mutex_ );

		while ( !queue_.empty( ) )
			queue_.pop( );
	}

private:
	template< typename... > friend class ::q::readable;

	template< typename Tuple >
	void _close( Tuple&& tup, bool force_exception = false )
	{
		shared_task notification;

		{
			Q_AUTO_UNIQUE_LOCK( mutex_ );

			bool was_closed = closed_.exchange(
				true, std::memory_order_seq_cst );

			// When the channel is a channel of promises, and the
			// channel is nicely closed, an inner promise that is
			// rejected should close the channel with an error.
			// In this case, we overwrite the non-error with this
			// error.
			bool has_exception = std::get< 0 >( close_exception_ );
			bool overwrite = !has_exception && force_exception;

			if ( !was_closed || overwrite )
				close_exception_ = std::forward< Tuple >( tup );

			for ( auto& waiter : waiters_ )
			{
				if ( std::get< 0 >( close_exception_ ) )
					waiter->set_exception( std::get< 1 >(
						close_exception_ ) );
				else
					waiter->set_closed( );
			}

			waiters_.clear( );

			scopes_.clear( );

			notification = resume_notification_;
		}

		if ( notification )
			notification( );
	}

	inline void resume( )
	{
		if ( paused_.exchange( false ) )
		{
			shared_task trigger_resume = resume_notification_;
			if ( trigger_resume )
				trigger_resume( );
		}
	}

	queue_ptr default_queue_;
	// TODO: Make this lock-free and consider other list types
	mutable mutex mutex_;
	std::list< std::unique_ptr< waiter_type > > waiters_;
	std::queue< tuple_type > queue_;
	// True if arbitrary exception, false if "closed exception"
	std::tuple< bool, std::exception_ptr > close_exception_;
	std::atomic< bool > closed_;
	std::atomic< bool > paused_;
	const std::size_t buffer_count_;
	const std::size_t resume_count_;
	shared_task resume_notification_;
	std::vector< scope > scopes_;
};

template< typename... T >
class shared_channel_owner
{
public:
	shared_channel_owner( ) = delete;
	shared_channel_owner( const shared_channel_owner& ) = default;
	shared_channel_owner( shared_channel_owner&& ) = default;

	shared_channel_owner& operator=( const shared_channel_owner& ) = default;
	shared_channel_owner& operator=( shared_channel_owner&& ) = default;

	shared_channel_owner(
		std::shared_ptr< detail::shared_channel< T... > > ch )
	: shared_channel_( std::move( ch ) )
	{ }

	~shared_channel_owner( )
	{
		shared_channel_->close( );
	}

protected:
	std::shared_ptr< detail::shared_channel< T... > > shared_channel_;
};

} // namespace detail

typedef options< concurrency > consume_options;

template< typename... T >
class readable
{
public:
	typedef typename detail::channel_traits< T... >::type traits;

	using is_promise = typename traits::is_promise;
	using promise_type = typename traits::promise_type;
	using unique_promise_type = typename traits::unique_promise_type;
	using tuple_type = typename traits::inner_tuple_type;
	using promise_arguments_type = typename traits::promise_arguments_type;

	readable( ) = default;
	readable( const readable& ) = default;
	readable( readable&& ) = default;

	readable& operator=( const readable& ) = default;
	readable& operator=( readable&& ) = default;

	template< bool IsPromise = is_promise::value >
	Q_NODISCARD
	typename std::enable_if<
		!IsPromise,
		unique_promise_type
	>::type
	read( )
	{
		return shared_channel_->read( );
	}

	template< bool IsPromise = is_promise::value >
	Q_NODISCARD
	typename std::enable_if<
		IsPromise,
		unique_promise_type
	>::type
	read( )
	{
		auto shared_channel = shared_channel_;

		return maybe_unshare( shared_channel_->read( )
		.then( [ shared_channel ](
			promise_type&& promise
		)
		{
			return promise
			.fail( [ shared_channel ]( std::exception_ptr e )
			-> promise_type
			{
				shared_channel->_close( std::make_tuple(
					true, e ), true );
				shared_channel->clear( );
				std::rethrow_exception(
					shared_channel->get_exception( ) );
			} );
		} ) );
	}

	template<
		typename FnValue,
		typename FnClosed,
		bool IsPromise = is_promise::value
	>
	Q_NODISCARD
	typename std::enable_if<
		detail::shared_channel< T... >
			::template fast_waiter_type_traits<
				decayed_function_t< FnValue >,
				decayed_function_t< FnClosed >
			>
			::inner_callbacks_are_valid::value
		and
		!IsPromise,
		promise< bool >
	>::type
	read( FnValue&& fn_value, FnClosed&& fn_closed )
	{
		return shared_channel_->read(
			std::forward< FnValue >( fn_value ),
			std::forward< FnClosed >( fn_closed )
		);
	}

	template<
		typename FnValue,
		typename FnClosed,
		bool IsPromise = is_promise::value
	>
	Q_NODISCARD
	typename std::enable_if<
		detail::shared_channel< T... >
			::template fast_waiter_type_traits<
				decayed_function_t< FnValue >,
				decayed_function_t< FnClosed >
			>
			::inner_callbacks_are_valid::value
		and
		IsPromise,
		promise< bool >
	>::type
	read( FnValue&& fn_value, FnClosed&& fn_closed )
	{
		auto shared_channel = shared_channel_;

		auto on_data = std::forward< FnValue >( fn_value );
		Q_MOVE_INTO_MOVABLE( on_data );

		auto on_value =
			[ shared_channel, Q_MOVABLE_MOVE( on_data ) ]
			( promise_type&& promise )
		mutable -> ::q::promise< >
		{
			return promise
			.then( Q_MOVABLE_CONSUME( on_data ) )
			.fail(
				[ shared_channel ]( std::exception_ptr e )
				mutable -> void
			{
				shared_channel->_close( std::make_tuple(
					true, e ), true );
				shared_channel->clear( );
				std::rethrow_exception(
					shared_channel->get_exception( ) );
			} );
		};

		return shared_channel_->read(
			std::move( on_value ),
			std::forward< FnClosed >( fn_closed )
		);
	}

	template< typename Fn >
	Q_NODISCARD
	typename std::enable_if<
		detail::shared_channel< T... >
			::template fast_waiter_type_traits<
				decayed_function_t< Fn >,
				function< void( ) >
			>
			::inner_callbacks_are_valid::value,
		promise< >
	>::type
	consume( Fn&& fn, consume_options options = consume_options( ) )
	{
		readable< T... > self = *this;

		auto _fn = decay_function( fn );
		auto _concurrency = options.get< concurrency >( ).get( );

		auto counter =
			std::make_shared< concurrency_counter >(
				get_queue( ), _concurrency );

		auto cb =
			[ self, _fn ]
			( resolver< > resolve, rejecter< > reject )
			mutable
		{
			typedef function< promise< >( ) > recurser_type;
			auto recurser = std::make_shared< recurser_type >( );

			auto completer = [ recurser, resolve ]( ) mutable
			{
				( *recurser ) = recurser_type( );
				resolve( );
			};

			auto failer =
				[ recurser, reject ]
				( std::exception_ptr err )
				mutable
			{
				( *recurser ) = recurser_type( );
				reject( std::move( err ) );
			};

			auto recurser_fn =
				[ self, _fn, recurser, completer, failer ]
				( )
				mutable
			{
				return self.read( _fn, completer )
				.then( [ self, recurser ]( bool got_data )
				mutable
				{
					if ( got_data )
						return ( *recurser )( );
					else
						return q::with(
							self.get_queue( ) );
				} )
				.fail( failer );
			};

			*recurser = std::move( recurser_fn );

			ignore_result( ( *recurser )( ) );
		};

		auto cb_concurrent =
			[ self, _fn, counter ]
			( resolver< > resolve, rejecter< > reject )
			mutable
		{
			typedef function< promise< >( ) >
				function_type;

			struct recurse_type
			{
				mutex mut_;
				function_type fn_;

				function_type get( )
				{
					Q_AUTO_UNIQUE_LOCK( mut_ );
					return fn_;
				}

				void clear( )
				{
					auto tmp = get( );
					{
						Q_AUTO_UNIQUE_LOCK( mut_ );
						fn_ = function_type( );
					}
				}
			};

			struct completion_type
			{
				enum class type
				{
					pending = 0,
					complete = 1,
					failed = 2,
				};

				std::atomic< type > type_;
				std::atomic< bool > has_exception_;
				std::exception_ptr exception_;
				resolver< > resolve_;
				rejecter< > reject_;

				completion_type(
					resolver< > resolve, rejecter< > reject
				)
				: type_( type::pending )
				, has_exception_( false )
				, exception_( )
				, resolve_( std::move( resolve ) )
				, reject_( std::move( reject ) )
				{ }

				void register_resolved( )
				{
					auto expected = type::pending;
					type_.compare_exchange_strong(
						expected, type::complete );
				}

				void register_rejected( std::exception_ptr err )
				{
					bool expected_err = false;
					auto changed = has_exception_
						.compare_exchange_strong(
							expected_err, true );

					if ( !changed )
						return;

					exception_ = err;

					auto expected = type::pending;
					type_.compare_exchange_strong(
						expected, type::complete );
				}

				void try_complete( )
				{
					type _type = type_;

					if ( _type == type::pending )
						return;
					else if ( _type == type::complete )
						resolve_( );
					else if ( _type == type::failed )
						reject_( exception_ );
				}
			};

			auto recurser = std::make_shared< recurse_type >( );
			auto end = std::make_shared< completion_type >(
				resolve, reject );

			counter->set_zero_function( [ end ]( )
			{
				end->try_complete( );
			} );

			auto completer = [ recurser, end, counter ]( ) mutable
			{
				end->register_resolved( );
				recurser->clear( );
				if ( counter->get( ) == 0 )
					end->try_complete( );
			};

			auto failer =
				[ recurser, end, counter ]
				( std::exception_ptr err )
				mutable
			{
				end->register_rejected( std::move( err ) );
				recurser->clear( );
				if ( counter->get( ) == 0 )
					end->try_complete( );
			};

			auto recurser_fn =
				[
					self,
					_fn,
					counter,
					recurser,
					completer,
					failer
				]
				( )
				mutable
			{
				queue_ptr queue = self.get_queue( );

				auto on_value =
				[ _fn, counter, failer, queue ]
				( tuple_type&& value )
				mutable
				{
					counter->inc( );

					q::with( queue, std::move( value ) )
					.then( std::move( _fn ) )
					.fail( failer )
					.finally( [ counter ]( )
					{
						counter->dec( );
					} );

					return counter->get_promise( );
				};

				return self.read( on_value, completer )
				.then( [ queue, recurser ]( bool got_data )
				mutable
				{
					if ( got_data )
					{
						auto recurse = recurser->get( );
						if ( recurse )
							return recurse( );
					}

					return q::with( queue );
				} )
				.fail( failer );
			};

			recurser->fn_ = std::move( recurser_fn );

			ignore_result( recurser->get( )( ) );
		};

		if ( _concurrency == 1 )
			return q::make_promise(
				get_queue( ), std::move( cb ) );
		else
			return q::make_promise(
				get_queue( ), std::move( cb_concurrent ) );
	}

	template< typename... U >
	typename std::enable_if<
		detail::pipe_helper< arguments< T... >, arguments< U... > >
			::valid::value
	>::type
	pipe( writable< U... > writable )
	{
		detail::pipe_helper< arguments< T... >, arguments< U... > >
			::pipe( *this, writable );
	}

	Q_NODISCARD
	std::size_t buffer_count( ) const
	{
		return shared_channel_->buffer_count( );
	}

	Q_NODISCARD
	bool is_closed( ) const
	{
		return shared_channel_->is_closed( );
	}

	Q_NODISCARD
	std::exception_ptr get_exception( ) const
	{
		return shared_channel_->get_exception( );
	}

	void close( )
	{
		shared_channel_->close( );
	}

	// Existence of this equals that of the shared_channel
	template< typename E >
	void close( E&& e )
	{
		shared_channel_->close( std::forward< E >( e ) );
	}

	void add_scope_until_closed( scope&& scope )
	{
		shared_channel_->add_scope_until_closed( std::move( scope ) );
	}

	Q_NODISCARD
	const queue_ptr& get_queue( ) const
	{
		return shared_channel_->get_queue( );
	}

	/**
	 * Clears this channel from any queued values. The channel can still be
	 * used for future values.
	 *
	 * NOTE: This will not trigger back pressure notifications! Use `clear`
	 *       with caution, as you can easily end up with a stuck channel.
	 *       Most likely you do not want to use this function to begin
	 *       with, as it is a grand anti-pattern, only useful in rare
	 *       situations.
	 */
	void clear( )
	{
		shared_channel_->clear( );
	}

private:
	readable( std::shared_ptr< detail::shared_channel< T... > > ch )
	: shared_channel_( ch )
	, shared_owner_(
		std::make_shared< detail::shared_channel_owner< T... > >( ch ) )
	{ }

	template< typename Promise, bool Shared = Promise::shared_type::value >
	typename std::enable_if<
		!Shared,
		typename Promise::unique_this_type
	>::type
	maybe_unshare( Promise&& promise )
	{
		return std::forward< Promise >( promise );
	}

	template< typename Promise, bool Shared = Promise::shared_type::value >
	typename std::enable_if<
		Shared,
		typename Promise::unique_this_type
	>::type
	maybe_unshare( Promise&& promise )
	{
		return promise.unshare( );
	}

	friend class channel< T... >;

	std::shared_ptr< detail::shared_channel< T... > > shared_channel_;
	std::shared_ptr< detail::shared_channel_owner< T... > > shared_owner_;
};

template< typename... T >
class writable
{
public:
	typedef typename detail::channel_traits< T... >::type traits;
	typedef typename traits::tuple_type tuple_type;
	typedef typename traits::is_promise is_promise;
	typedef typename traits::promise_type promise_type;
	typedef typename traits::promise_tuple_type promise_tuple_type;
	typedef typename traits::outer_tuple_type outer_tuple_type;
	typedef typename traits::promise_arguments_type promise_arguments_type;

	writable( ) = default;
	writable( const writable& ) = default;
	writable( writable&& ) = default;

	writable& operator=( const writable& ) = default;
	writable& operator=( writable&& ) = default;

	/**
	 * ( tuple< T... > ) ->
	 */
	template< typename Tuple >
	Q_NODISCARD
	typename std::enable_if<
		q::is_tuple< typename std::decay< Tuple >::type >::value
		and
		tuple_arguments_t<
			typename std::decay< Tuple >::type
		>::template is_convertible_to<
			tuple_arguments_t< outer_tuple_type >
		>::value,
		bool
	>::type
	write( Tuple&& t )
	{
		return shared_channel_->write( std::forward< Tuple >( t ) );
	}

	/**
	 * ( tuple< void_t > ) -> tuple< > (stripped from void_t)
	 */
	template< typename Tuple >
	Q_NODISCARD
	typename std::enable_if<
		q::is_tuple< typename std::decay< Tuple >::type >::value
		and
		std::is_same<
			typename std::decay< Tuple >::type,
			std::tuple< void_t >
		>::value
		and
		std::is_same<
			outer_tuple_type,
			std::tuple< >
		>::value,
		bool
	>::type
	write( Tuple&& )
	{
		return write( std::make_tuple( ) );
	}

	/**
	 * ( Args... ) -> tuple< T... >
	 */
	template< typename... Args >
	Q_NODISCARD
	typename std::enable_if<
		(
			arguments<
				typename std::decay< Args >::type...
			>::template is_convertible_to_incl_void<
				arguments< T... >
			>::value
		)
		or
		(
			is_promise::value
			and
			(
				arguments<
					typename std::decay< Args >::type...
				>::template is_convertible_to_incl_void<
					arguments< promise_type >
				>::value
				or
				arguments<
					typename std::decay< Args >::type...
				>::template is_convertible_to_incl_void<
					typename promise_type::argument_types
				>::value
			)
		),
		bool
	>::type
	write( Args&&... args )
	{
		return write(
			std::make_tuple( std::forward< Args >( args )... ) );
	}

	/**
	 * Promise-channel:
	 * ( tuple< void_t > ) -> tuple< >
	 * (stripped from void_t)
	 */
	template< typename Tuple >
	Q_NODISCARD
	typename std::enable_if<
		is_promise::value
		and
		std::is_same<
			typename std::decay< Tuple >::type,
			std::tuple< void_t >
		>::value
		and
		std::is_same<
			promise_arguments_type,
			q::arguments< >
		>::value,
		bool
	>::type
	write( Tuple&& )
	{
		return write( std::make_tuple( ) );
	}

	/**
	 * Promise-channel:
	 * ( tuple< T... > ) -> tuple< promise< T... > >
	 */
	template< typename Tuple >
	Q_NODISCARD
	typename std::enable_if<
		is_promise::value
		and
		tuple_arguments_t< typename std::decay< Tuple >::type >
			::template is_convertible_to< promise_arguments_type >
			::value,
		bool
	>::type
	write( Tuple&& t )
	{
		return write( std::make_tuple( suitable_with(
			shared_channel_->get_queue( ),
			std::forward< Tuple >( t )
		) ) );
	}

	/**
	 * Like write() but throws q::channel_closed_exception if the channel
	 * was closed.
	 */
	template< typename... Any >
	void ensure_write( Any&&... t )
	{
		if ( !write( std::forward< Any >( t )... ) )
			Q_THROW( channel_closed_exception( ) );
	}

	Q_NODISCARD
	bool should_write( ) const
	{
		return shared_channel_->should_write( );
	}

	void set_resume_notification( shared_task fn, bool trigger_now = false )
	{
		shared_channel_->set_resume_notification(
			std::move( fn ), trigger_now );
	}

	void unset_resume_notification( )
	{
		shared_channel_->unset_resume_notification( );
	}

	void trigger_resume_notification( )
	{
		shared_channel_->trigger_resume_notification( );
	}

	Q_NODISCARD
	bool is_closed( ) const
	{
		return shared_channel_->is_closed( );
	}

	Q_NODISCARD
	bool has_exception( ) const
	{
		return shared_channel_->has_exception( );
	}

	Q_NODISCARD
	std::exception_ptr get_exception( ) const
	{
		return shared_channel_->get_exception( );
	}

	void close( )
	{
		shared_channel_->close( );
	}

	// Existence of this equals that of the shared_channel
	template< typename E >
	void close( E&& e )
	{
		shared_channel_->close( std::forward< E >( e ) );
	}

	void add_scope_until_closed( scope&& scope )
	{
		shared_channel_->add_scope_until_closed( std::move( scope ) );
	}

	Q_NODISCARD
	const queue_ptr& get_queue( ) const
	{
		return shared_channel_->get_queue( );
	}

private:
	writable( std::shared_ptr< detail::shared_channel< T... > > ch )
	: shared_channel_( ch )
	, shared_owner_(
		std::make_shared< detail::shared_channel_owner< T... > >( ch ) )
	{ }

	template< typename Tuple, bool Shared = traits::is_shared::value >
	typename std::enable_if< !Shared, promise_type >::type
	suitable_with( const queue_ptr& queue, Tuple&& t )
	{
		return q::with( queue, std::forward< Tuple >( t ) );
	}

	template< typename Tuple, bool Shared = traits::is_shared::value >
	typename std::enable_if< Shared, promise_type >::type
	suitable_with( const queue_ptr& queue, Tuple&& t )
	{
		return q::with( queue, std::forward< Tuple >( t ) ).share( );
	}

	friend class channel< T... >;

	std::shared_ptr< detail::shared_channel< T... > > shared_channel_;
	std::shared_ptr< detail::shared_channel_owner< T... > > shared_owner_;
};

template< typename... T >
class channel
{
public:
	typedef typename detail::channel_traits< T... > traits;

	channel( const queue_ptr& queue, std::size_t buffer_count )
	: channel(
		queue,
		buffer_count,
		detail::default_resume_count( buffer_count )
	)
	{ }

	channel(
		const queue_ptr& queue,
		std::size_t buffer_count,
		std::size_t resume_count
	)
	: shared_channel_(
		q::make_shared< detail::shared_channel< T... > >(
			queue, buffer_count, resume_count ) )
	, readable_( shared_channel_ )
	, writable_( shared_channel_ )
	{ }

	readable< T... > get_readable( )
	{
		return readable_;
	}

	writable< T... > get_writable( )
	{
		return writable_;
	}

	void add_scope_until_closed( scope&& scope )
	{
		shared_channel_->add_scope_until_closed( std::move( scope ) );
	}

	const queue_ptr& get_queue( ) const
	{
		return shared_channel_->get_queue( );
	}

private:
	std::shared_ptr< detail::shared_channel< T... > > shared_channel_;
	readable< T... > readable_;
	writable< T... > writable_;
};

Q_MAKE_SIMPLE_EXCEPTION( internal_backpressure_exception );

/**
 * backpressure is a helper class to allow multiple writables to signal
 * upstream backpressure notifications, and when all have signaled, this class
 * emits a downstream event (through a promise).
 */
class backpressure
: public std::enable_shared_from_this< backpressure >
{
public:
	template< typename... T >
	void add_once( writable< T... > writable )
	{
		auto self = shared_from_this( );

		++counter_;

		writable.set_resume_notification(
			[ self, writable ]( ) mutable
			{
				self->handle_one( );
				writable.unset_resume_notification( );
			},
			true
		);
	}

	readable< > get_readable( )
	{
		return back_ch_.get_readable( );
	}

	promise< > get_promise( )
	{
		++counter_;

		auto promise = get_readable( ).read( );

		handle_one( );

		return promise;
	}

protected:
	backpressure( const queue_ptr& queue )
	: counter_( 0 )
	, back_ch_( queue, 3 )
	, writable_( back_ch_.get_writable( ) )
	{ }

	~backpressure( )
	{
		while ( counter_ > 0 )
			handle_one( );
	}

private:
	void handle_one( )
	{
		if ( !counter_ )
		{
			// Shouldn't happen
			writable_.close( internal_backpressure_exception( ) );
		}
		else if ( !--counter_ )
		{
			// All pending/incoming channels are listened to.
			// Forward one.
			ignore_result( writable_.write( ) );
		}
	}

	std::atomic< std::size_t > counter_;
	channel< > back_ch_;
	writable< > writable_;
};

} // namespace q

#endif // LIBQ_CHANNEL_HPP
