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

#ifndef LIBQ_PROMISE_SIGNAL_HPP
#define LIBQ_PROMISE_SIGNAL_HPP

#include <q/types.hpp>

#include <memory>

namespace q { namespace detail {

// TODO: Make lock-free with a lock-free queue and atomic bool.
class promise_signal
: public std::enable_shared_from_this< promise_signal >
{
public:
	~promise_signal( );

	void done( ) noexcept;

	void push( task&& task, const queue_ptr& queue ) noexcept;
	void push_synchronous( task&& task ) noexcept;

protected:
	promise_signal( );

private:
	struct pimpl;

	std::unique_ptr< pimpl > pimpl_;
};

typedef std::shared_ptr< promise_signal > promise_signal_ptr;

} } // namespace detail, namespace queue

#endif // LIBQ_PROMISE_SIGNAL_HPP
