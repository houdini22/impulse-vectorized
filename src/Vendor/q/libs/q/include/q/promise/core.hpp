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

#ifndef LIBQ_PROMISE_CORE_HPP
#define LIBQ_PROMISE_CORE_HPP

#include <q/exception.hpp>

namespace q {

template< typename... T >
class promise;

template< typename... T >
class shared_promise;

namespace detail {

template< typename... T > class defer;
template< bool, typename... > class generic_promise;

} // namespace detail

template< class T >
struct is_promise
: std::false_type
{ };

template< bool B, typename... T >
struct is_promise< detail::generic_promise< B, T... > >
: std::true_type
{ };

template< typename... T >
struct is_promise< promise< T... > >
: std::true_type
{ };

template< typename... T >
struct is_promise< shared_promise< T... > >
: std::true_type
{ };

#ifdef LIBQ_WITH_CPP14

template< typename... T >
constexpr bool is_promise_v = is_promise< T... >::value;

#endif // LIBQ_WITH_CPP14

// TODO: Make _t and _v
template< class... T >
struct are_promises
: fold_t<
	q::arguments< T... >,
	generic_operator<
		is_promise, logic_and
	>::template fold_type,
	std::true_type
>
{ };

template< >
struct are_promises< >
: std::false_type
{ };

namespace detail {

template< typename... T >
struct promise_if_first_and_only
{
	typedef std::false_type valid;
	typedef void type;
	typedef std::tuple< > tuple_type;
	typedef arguments< > arguments_type;
};

template< typename... Args >
struct promise_if_first_and_only< ::q::promise< Args... > >
{
	typedef std::true_type valid;
	typedef promise< Args... > type;
	typedef typename type::tuple_type tuple_type;
	typedef arguments< Args... > arguments_type;
};

template< typename... Args >
struct promise_if_first_and_only< ::q::shared_promise< Args... > >
{
	typedef std::true_type valid;
	typedef promise< Args... > type;
	typedef typename type::tuple_type tuple_type;
	typedef arguments< Args... > arguments_type;
};

} // namespace detail

class generic_combined_promise_exception
: public exception
{
public:
	generic_combined_promise_exception( ) = default;

	const std::vector< std::exception_ptr >& exceptions( ) const
	{
		return exceptions_;
	}

protected:
	void add_exception( const std::exception_ptr& e )
	{
		exceptions_.push_back( e );
	}

private:
	std::vector< std::exception_ptr > exceptions_;
};

template< typename T >
class combined_promise_exception
: public generic_combined_promise_exception
{
public:
	typedef std::vector< expect< T > > exception_type;

	combined_promise_exception( ) = delete;
	combined_promise_exception( std::vector< expect< T > >&& data )
	: data_( new exception_type( std::move( data ) ) )
	{
		for ( auto& element : *data_ )
			if ( element.has_exception( ) )
				add_exception( element.exception( ) );
	}

	const std::vector< expect< T > >& data( )
	{
		return *data_;
	}

	std::vector< expect< T > > consume( )
	{
		return std::move( *data_ );
	}

private:
	// The data needs to be shared_ptr'd as C++ internally may copy the data
	// when dealing with std::exception_ptr's.
	std::shared_ptr< exception_type > data_;
};

namespace detail {

template<
	typename T,
	bool B = is_promise< typename std::decay< T >::type >::value
>
struct argument_types_if_promise
{
	typedef typename std::decay< T >::type::argument_types types;
};

template< typename T >
struct argument_types_if_promise< T, false >
{
	typedef struct { } types;
};

template< typename T >
struct promise_arguments
{
	typedef ::q::arguments< T > type;
};

template< typename... T, bool B >
struct promise_arguments< generic_promise< B, T... > >
{
	typedef typename generic_promise< B, T... >::argument_types type;
};

template< typename... T >
struct promise_arguments< ::q::promise< T... > >
{
	typedef typename ::q::promise< T... >::argument_types type;
};

template< typename... T >
struct promise_arguments< ::q::shared_promise< T... > >
{
	typedef typename ::q::shared_promise< T... >::argument_types type;
};

} // namespace detail

template< typename T >
struct promise_arguments
: detail::promise_arguments< typename std::decay< T >::type >
{ };

template< typename... Promises >
struct merge_promise_arguments;

template< typename First, typename... Rest >
struct merge_promise_arguments< First, Rest... >
: std::conditional<
	sizeof...( Rest ) == 0,
	promise_arguments<
		typename std::decay< First >::type
	>,
	typename merge<
		arguments,
		typename promise_arguments<
			typename std::decay< First >::type
		>::type,
		typename promise_arguments<
			typename std::decay< Rest >::type
		>::type...
	>::type
>::type
{ };

template< typename Only >
struct merge_promise_arguments< Only >
: detail::argument_types_if_promise< Only >::types
{ };

template< >
struct merge_promise_arguments< >
: q::arguments< >
{ };

typedef options<
	queue_ptr,
	defaultable< queue_ptr >
> queue_options;

}

#endif // LIBQ_PROMISE_CORE_HPP
