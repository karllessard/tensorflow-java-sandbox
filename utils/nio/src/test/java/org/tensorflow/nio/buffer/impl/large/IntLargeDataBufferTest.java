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
package org.tensorflow.nio.buffer.impl.large;

import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.IntDataBufferTestBase;

public class IntLargeDataBufferTest extends IntDataBufferTestBase {

  @Override
  protected long maxCapacity() {
    return IntLargeDataBuffer.MAX_CAPACITY;
  }

  @Override
  protected IntDataBuffer allocate(long capacity) {
    return IntLargeDataBuffer.allocate(capacity);
  }
}
