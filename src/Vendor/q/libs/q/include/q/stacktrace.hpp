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

#ifndef LIBQ_STACKTRACE_HPP
#define LIBQ_STACKTRACE_HPP

#include <iosfwd>
#include <vector>
#include <string>

namespace q {

class stacktrace
{
public:
	struct frame
	{
		std::size_t frameno;
		std::string lib;
		std::string lib_path;
		std::uint64_t addr;
		std::string symbol;
		std::string extra;
	};

	stacktrace( ) = delete;
	stacktrace( stacktrace&& ) = default;
	stacktrace( const stacktrace& ) = delete;

	stacktrace( std::vector< frame >&& frames )
	noexcept;

	const std::vector< frame >& frames( ) const
	noexcept;

	std::string string( ) const
	noexcept;

private:
	std::vector< frame > frames_;
};

std::ostream& operator<<( std::ostream& os, const stacktrace& st )
noexcept;

typedef stacktrace( *stacktrace_function )( );

/**
 * Registers a function to be called when get_stacktrace() is invoked. This
 * affects how q creates stack traces.
 *
 * @param stacktrace_function The function to be used, or nullptr to reset to
 * the default built-in stack trace function.
 *
 * @returns the previous stacktrace function, or nullptr if the default
 * stacktrace function was in use.
 */
stacktrace_function register_stacktrace_function( stacktrace_function )
noexcept;

/**
 * Creates a stack trace on the current thread and returns it as a stacktrace
 * object.
 */
stacktrace get_stacktrace( )
noexcept;

} // namespace q

#endif // LIBQ_STACKTRACE_HPP
