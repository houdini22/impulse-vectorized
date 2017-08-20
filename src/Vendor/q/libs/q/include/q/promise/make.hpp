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

#ifndef LIBQ_PROMISE_MAKE_HPP
#define LIBQ_PROMISE_MAKE_HPP

namespace q {

template< typename... Args >
struct resolver
{
	resolver( ) = default;

	resolver( const std::shared_ptr< ::q::detail::defer< Args... > >& d )
	: deferred_( d )
	{ }

	template< typename... Args_ >
	typename std::enable_if<
		::q::arguments< Args_... >::template is_convertible_to<
			::q::arguments< Args... >
		>::value
	>::type
	operator( )( Args_... args )
	{
		deferred_->set_value( std::forward< Args >( args )... );
	}

private:
	std::shared_ptr< ::q::detail::defer< Args... > > deferred_;
};

template< typename... Args >
struct rejecter
{
	rejecter( ) = default;

	rejecter( const std::shared_ptr< ::q::detail::defer< Args... > >& d )
	: deferred_( d )
	{ }

	template< typename E >
	void operator( )( E&& e )
	{
		deferred_->set_exception( std::forward< E >( e ) );
	}

private:
	std::shared_ptr< ::q::detail::defer< Args... > > deferred_;
};

namespace detail {

template< typename... Args >
struct resolve_helper_
: std::enable_shared_from_this< resolve_helper_< Args... > >
{
	resolve_helper_( const ::q::queue_ptr& queue )
	: deferred_( q::detail::defer< Args... >::construct( queue ) )
	, resolver_( deferred_ )
	, rejecter_( deferred_ )
	{ }

	template< typename Fn >
	void run( Fn&& fn )
	{
		try
		{
			fn( std::move( resolver_ ), std::move( rejecter_ ) );
		}
		catch ( ... )
		{
			deferred_->set_exception( std::current_exception( ) );
		}
	}

	::q::promise< Args... > get_promise( ) const
	{
		return deferred_->get_promise( );
	}

private:
	std::shared_ptr< ::q::detail::defer< Args... > > deferred_;
	resolver< Args... > resolver_;
	rejecter< Args... > rejecter_;
};

template< typename... Args >
struct resolve_helper
: resolve_helper_< typename ::q::remove_rvalue_reference< Args >::type... >
{
	typedef resolve_helper_<
		typename ::q::remove_rvalue_reference< Args >::type...
	> base_type;

	resolve_helper( const ::q::queue_ptr& queue )
	: base_type( queue )
	{ }
};

} // namespace detail

template< typename Fn >
typename std::enable_if<
	arity_of_t< Fn >::value == 0,
	detail::suitable_promise_t< result_of_as_tuple_t< Fn > >
>::type
make_promise( const queue_ptr& queue, Fn&& fn )
{
	auto deferred = result_of_as_argument_t< Fn >
		::template apply< ::q::detail::defer >
		::construct( queue );

	Q_MAKE_MOVABLE( fn );

	queue->push( [ deferred, Q_MOVABLE_FORWARD( fn ) ]( ) mutable
	{
		deferred->set_by_fun( Q_MOVABLE_CONSUME( fn ) );
	} );

	return deferred->get_promise( );
}

template< typename Fn >
typename std::enable_if<
	arity_of_t< Fn >::value == 0,
	detail::suitable_promise_t< result_of_as_tuple_t< Fn > >
>::type
make_promise_sync( const queue_ptr& queue, Fn&& fn )
{
	auto deferred = result_of_as_argument_t< Fn >
		::template apply< ::q::detail::defer >
		::construct( queue );

	deferred->set_by_fun( std::forward< Fn >( fn ) );

	return deferred->get_promise( );
}

template< typename T >
struct extract_resolver_args;

template< typename... Args >
struct extract_resolver_args< ::q::resolver< Args... > >
{
	typedef ::q::arguments< Args... > type;
};

template< typename Fn >
typename std::enable_if<
	true,
	detail::suitable_promise_t<
		typename extract_resolver_args<
			typename std::decay<
				first_argument_of_t< Fn >
			>::type
		>::type::tuple_type
	>
>::type
make_promise( const ::q::queue_ptr& queue, Fn&& fn )
{
	typedef first_argument_of_t< Fn > resolve_signature;
	typedef typename extract_resolver_args<
		typename std::decay< resolve_signature >::type
	>::type                                  value_types;
	typedef typename value_types::template apply<
		detail::resolve_helper
	>                                        resolve_helper;

	auto helper = std::make_shared< resolve_helper >( queue );

	Q_MAKE_MOVABLE( fn );

	queue->push( [ helper, Q_MOVABLE_FORWARD( fn ) ]( ) mutable
	{
		helper->run( Q_MOVABLE_CONSUME( fn ) );
	} );

	return helper->get_promise( );
}

template< typename Fn >
typename std::enable_if<
	true,
	detail::suitable_promise_t<
		typename extract_resolver_args<
			typename std::decay<
				first_argument_of_t< Fn >
			>::type
		>::type::tuple_type
	>
>::type
make_promise_sync( const ::q::queue_ptr& queue, Fn&& fn )
{
	typedef first_argument_of_t< Fn > resolve_signature;
	typedef typename extract_resolver_args<
		typename std::decay< resolve_signature >::type
	>::type                                  value_types;
	typedef typename value_types::template apply<
		detail::resolve_helper
	>                                        resolve_helper;

	auto helper = std::make_shared< resolve_helper >( queue );

	helper->run( std::forward< Fn >( fn ) );

	return helper->get_promise( );
}

#ifdef LIBQ_WITH_CPP14

template< typename... Args, typename Fn >
q::promise< typename q::remove_rvalue_reference< Args >::type... >
make_promise_of( const ::q::queue_ptr& queue, Fn&& fn )
{
	typedef std::tuple<
		typename q::remove_rvalue_reference< Args >::type...
	> tuple_type;

	Q_MAKE_MOVABLE( fn );

	auto deferred = q::detail::defer< tuple_type >::construct( queue );

	queue->push( [ deferred, Q_MOVABLE_FORWARD( fn ) ]( ) mutable
	{
		auto resolve = [ deferred ]( Args&&... args )
		{
			deferred->set_value( std::forward< Args >( args )... );
		};

		auto reject = [ deferred ]( auto e )
		{
			deferred->set_exception( Q_FORWARD( e ) );
		};

		try
		{
			Q_MOVABLE_CONSUME( fn )( resolve, reject );
		}
		catch ( ... )
		{
			deferred->set_exception( std::current_exception( ) );
		}

	} );

	return deferred->get_promise( );
}

template< typename... Args, typename Fn >
q::promise< typename q::remove_rvalue_reference< Args >::type... >
make_promise_sync_of( const ::q::queue_ptr& queue, Fn&& fn )
{
	typedef std::tuple<
		typename q::remove_rvalue_reference< Args >::type...
	> tuple_type;

	auto deferred = q::detail::defer< tuple_type >::construct( queue );

	auto resolve = [ deferred ]( Args&&... args )
	{
		deferred->set_value( std::forward< Args >( args )... );
	};

	auto reject = [ deferred ]( auto e )
	{
		deferred->set_exception( Q_FORWARD( e ) );
	};

	try
	{
		fn( resolve, reject );
	}
	catch ( ... )
	{
		deferred->set_exception( std::current_exception( ) );
	}

	return deferred->get_promise( );
}

#endif

} // namespace q

#endif // LIBQ_PROMISE_MAKE_HPP
