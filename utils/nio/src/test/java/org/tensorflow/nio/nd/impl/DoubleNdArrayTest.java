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
package org.tensorflow.nio.nd.impl;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrayTestBase;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;

public class DoubleNdArrayTest extends NdArrayTestBase<Double> {

  @Override
  protected NdArray<Double> allocate(Shape shape) {
    return NdArrays.ofDoubles(shape);
  }

  @Override
  protected DataBuffer<Double> allocateBuffer(long capacity) {
    return DataBuffers.ofDoubles(capacity);
  }

  @Override
  protected Double valueOf(Long val) {
    return val.doubleValue();
  }
}
