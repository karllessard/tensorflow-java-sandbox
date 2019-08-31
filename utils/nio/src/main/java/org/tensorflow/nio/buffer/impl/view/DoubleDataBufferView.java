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
package org.tensorflow.nio.buffer.impl.view;

import java.util.stream.DoubleStream;

import org.tensorflow.nio.buffer.DoubleDataBuffer;

public class DoubleDataBufferView extends DataBufferView<Double, DoubleDataBuffer> implements
    DoubleDataBuffer {

  public DoubleDataBufferView(DoubleDataBuffer delegate, long start, long end) {
    super(delegate, start, end);
  }

  @Override
  public DoubleStream doubleStream() {
    // TODO
    throw new UnsupportedOperationException();
  }

  @Override
  public DoubleDataBuffer get(double[] dst, int offset, int length) {
    return delegate.get(dst, offset, length);
  }

  @Override
  public DoubleDataBuffer put(double[] src, int offset, int length) {
    return delegate.put(src, offset, length);
  }

  @Override
  public DoubleDataBuffer duplicate() {
    return new DoubleDataBufferView(delegate.duplicate(), start, end);
  }

  @Override
  public DoubleDataBuffer slice() {
    return new DoubleDataBufferView(delegate.duplicate(), position(), limit());
  }
}
