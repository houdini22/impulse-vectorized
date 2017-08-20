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

#ifndef LIBQ_SCOPE_HPP
#define LIBQ_SCOPE_HPP

#include <q/function.hpp>

namespace q {

namespace detail {

struct holder
{
	virtual ~holder( )
	{ }
};

template< typename T >
struct deleter
: holder
{
	deleter( T&& t )
	: holder_( q::make_unique< T >( std::move( t ) ) )
	{ }

	~deleter( )
	{ }

	std::unique_ptr< T > holder_;
};

typedef std::unique_ptr< holder > holder_ptr;

} // namespace detail

class scope
{
public:
	scope( ) = delete;
	scope( const scope& ) = delete;
	scope( scope&& ) = default;

	scope( detail::holder_ptr&& holder )
	: holder_( std::move( holder ) )
	{ }

	~scope( )
	{ }

	scope& operator=( scope&& ) = default;
	scope& operator=( const scope& ) = delete;

private:
	detail::holder_ptr holder_;
};

template< typename T >
typename std::enable_if<
	!std::is_lvalue_reference< T >::value
	and
	!std::is_const< T >::value,
	scope
>::type
make_scope( T&& t )
{
	typedef typename std::decay< T >::type type;

	return scope( make_unique< detail::deleter< type > >( std::move( t ) ) );
}

template< typename T >
typename std::enable_if<
	std::is_lvalue_reference< T >::value
	and
	is_shared_pointer< typename std::decay< T >::type >::value,
	scope
>::type
make_scope( T&& t )
{
	typedef typename std::decay< T >::type type;

	return scope( make_unique< detail::deleter< type > >( type( t ) ) );
}

class scoped_function
{
public:
	typedef q::unique_function< void( ) > function_type;

	scoped_function( ) = delete;
	scoped_function( scoped_function&& ref )
	: fn_( std::move( ref.fn_ ) )
	{ }
	scoped_function( const scoped_function& ) = delete;

	template< typename Fn >
	scoped_function( Fn&& fn )
	: fn_( std::forward< Fn >( fn ) )
	{ }

	~scoped_function( )
	{
		// TODO: Handle uncaught exceptions here
		if ( fn_ )
			fn_( );
	}

private:
	function_type fn_;
};

namespace detail {

template< typename Fn >
struct decayed_function
{
	typedef typename std::remove_reference<
		typename std::decay< Fn >::type
	>::type type;
};

} // namespace detail

template< typename Fn >
typename std::enable_if<
	Q_ARITY_OF( Fn ) == 0 &&
	Q_RESULT_OF_AS_ARGUMENT( Fn )::size::value == 0,
	scope
>::type
make_scoped_function( Fn&& fn )
{
	return make_scope( scoped_function( std::forward< Fn >( fn ) ) );
}

} // namespace q

#endif // LIBQ_SCOPE_HPP
