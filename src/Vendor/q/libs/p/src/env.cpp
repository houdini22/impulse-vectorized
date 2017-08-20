
#include <p/env.hpp>

#include <cstdlib>

namespace p {

struct environment::pimpl
{
	;
};

environment::environment( )
: pimpl_( new pimpl )
{ }
environment::~environment( )
{ }


std::string environment::holder::string( ) const
{
	return std::getenv( name_.c_str( ) );
}

environment::holder::operator std::string( ) const
{
	return string( );
}

environment::holder& environment::holder::operator=( std::string value )
{
	::setenv( name_.c_str( ), value.c_str( ), 1 );
	return *this;
}

environment::holder::holder( environment& env, std::string&& name )
: env_( env )
{ }

} // namespace p
