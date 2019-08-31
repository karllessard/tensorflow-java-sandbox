/*
 Copyright 2019 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.nio.nd;

import java.util.Iterator;

/**
 * An iterator that allows read and/or write operations over a sequence of values.
 *
 * @param <T> type of value
 */
public interface ValueIterator<T> extends Iterator<T> {

  /**
   * Sets the value at the next position of this iterator and increment it.
   *
   * <p>This additional method to the {@link Iterator} interface allows initializing the values
   * in a sequence. For example:
   * <pre>{@code
   * long value = 0L;
   * for (ValueIterator<Long> iter = array.values().iterator(); iter.hasNext();) {
   *     iter.next(value++);
   * }
   * }</pre>
   *
   * @param value value to set
   */
  void next(T value);
}
