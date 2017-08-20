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

#include <q/pp.hpp>

#ifdef LIBQ_ON_LINUX

#include "stacktrace.hpp"

#include <cstring>

namespace q {

namespace detail {

/*
 * Format:
 /<path>/q/libq.so(_ZN1q14get_stacktraceEv+0xc4) [0x7fb31c75cdb4]
 */
stacktrace::frame parse_stack_frame( const char* data )
noexcept
{
	stacktrace::frame frame;

	const char* pos, *pos2;

	pos = data;
	pos2 = std::strchr( pos, '(' );
	if ( pos2 )
	{
		// On Linux, backtrace_symbols return string is modifyable
		const_cast< char& >( pos2[ 0 ] ) = 0;
		auto names = get_file_name( pos );
		frame.lib = std::move( names.first );
		frame.lib_path = std::move( names.second );
		pos = pos2 + 1;
	}
	else
		pos = nullptr;

	if ( pos )
	{
		pos2 = pos + std::strcspn( pos, "+)" );

		frame.symbol.append( pos, pos2 - pos );

		if ( *pos2 == '+' )
		{
			// Extra (offset) available
			pos = pos2 + 1;
			pos2 = std::strchr( pos, ')' );
			if ( pos2 )
			{
				frame.extra.append( pos, pos2 - pos );
				pos = pos2 + 1;
			}
			else
			{
				frame.extra.append( pos );
				pos = nullptr;
			}
		}
		else
			pos = pos2 + 1;
	}

	if ( pos )
		pos = std::strchr( pos, '[' );

	if ( pos )
	{
		pos += 1;
		pos2 = std::strchr( pos, ']' );

		if ( pos[ 0 ] == '0' && pos[ 1 ] == 'x' )
			frame.addr = hex_to_dec( pos + 2 );

		pos = pos2;
	}

	return frame;
}

}

}

#endif
