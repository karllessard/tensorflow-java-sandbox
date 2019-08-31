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
package org.tensorflow.nio.buffer.impl.single;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBufferTestBase;

public class StringArrayDataBufferTest extends DataBufferTestBase<String> {

  @Override
  protected long maxCapacity() {
    return ArrayDataBuffer.MAX_CAPACITY;
  }

  @Override
  protected DataBuffer<String> allocate(long capacity) {
    return ArrayDataBuffer.allocate(String.class, capacity);
  }

  @Override
  protected String valueOf(Long val) {
    return val.toString();
  }
}
