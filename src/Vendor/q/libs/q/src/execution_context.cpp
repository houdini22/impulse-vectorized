
#include <q/execution_context.hpp>
#include <q/scheduler.hpp>

namespace q {

struct execution_context::pimpl
{
	event_dispatcher_ptr event_dispatcher_;
	queue_ptr queue_;
	scheduler_ptr scheduler_;
};

execution_context::execution_context(
	event_dispatcher_ptr ed,
	const scheduler_ptr& s )
: pimpl_( new pimpl )
{
	pimpl_->event_dispatcher_ = ed;
	pimpl_->queue_ = q::make_shared< q::queue >( 0 );
	pimpl_->scheduler_ = s;

	pimpl_->scheduler_->add_queue( pimpl_->queue_ );
}

execution_context::~execution_context( )
{ }

queue_ptr execution_context::queue( ) const
{
	return pimpl_->queue_;
}

scheduler_ptr execution_context::scheduler( ) const
{
	return pimpl_->scheduler_;
}

} // namespace q
