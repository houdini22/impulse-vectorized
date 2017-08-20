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

#ifndef LIBQ_PROMISE_HPP
#define LIBQ_PROMISE_HPP

#include <q/functional.hpp>
#include <q/log.hpp>
#include <q/temporarily_copyable.hpp>
#include <q/queue.hpp>
#include <q/exception.hpp>
#include <q/expect.hpp>
#include <q/memory.hpp>
#include <q/set_default.hpp>
#include <q/options.hpp>

#include <q/promise/async_task.hpp>
#include <q/promise/core.hpp>
#include <q/promise/signal.hpp>
#include <q/promise/state.hpp>
#include <q/promise/promise.hpp>
#include <q/promise/defer.hpp>
#include <q/promise/reject.hpp>
#include <q/promise/with.hpp>
#include <q/promise/all.hpp>
#include <q/promise/make.hpp>
#include <q/promise/delay.hpp>
#include <q/promise/promisify.hpp>
#include <q/promise/impl/then.hpp>
#include <q/promise/impl/fail.hpp>
#include <q/promise/impl/finally.hpp>
#include <q/promise/impl/tap.hpp>
#include <q/promise/impl/tap_error.hpp>
#include <q/promise/impl/rest.hpp>

namespace q {

// TODO: Get rid of std::promise <-> std::future as we manually encapsulate
// data and exceptions in q::expect and signal manually too.

} // namespace q

#endif // LIBQ_PROMISE_HPP
