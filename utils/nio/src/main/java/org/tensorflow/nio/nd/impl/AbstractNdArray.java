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
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.iterator.Iterators;
import org.tensorflow.nio.nd.ValueIterable;
import org.tensorflow.nio.nd.ValueIterator;

@SuppressWarnings("unchecked")
public abstract class AbstractNdArray<T, U extends NdArray<T>> implements NdArray<T> {

  @Override
  public Shape shape() {
    return shape;
  }

  @Override
  public long size() {
    return shape().size();
  }

  @Override
  public ValueIterable<T> values() {
    return Iterators.valuesOf(this);
  }

  @Override
  public Iterable<U> childElements() {
    return (Iterable) (() -> Iterators.elementsOf(this));
  }

  @Override
  public U read(T[] dst) {
    return (U) read(DataBuffers.wrap(dst, false));
  }

  @Override
  public U read(T[] dst, int offset) {
    return (U) read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public U write(T[] src) {
    return (U) write(DataBuffers.wrap(src, false));
  }

  @Override
  public U write(T[] src, int offset) {
    return (U) write(DataBuffers.wrap(src, false).position(offset));
  }

  protected AbstractNdArray(Shape shape) {
    this.shape = shape;
  }

  protected void slowRead(DataBuffer<T> buffer) {
    values().iterator().forEachRemaining(buffer::put);
  }

  protected void slowWrite(DataBuffer<T> buffer) {
    for (ValueIterator<T> dstIter = values().iterator(); dstIter.hasNext(); ) {
      dstIter.next(buffer.get());
    }
  }

  protected static <T> void slowCopy(NdArray<T> src, NdArray<T> dst) {
    if (!src.shape().equals(dst.shape())) {
      throw new IllegalArgumentException("Can only copy to arrays of the same shape");
    }
    for (ValueIterator<T> srcIter = src.values().iterator(), dstIter = dst.values().iterator();
        srcIter.hasNext(); ) {
      dstIter.next(srcIter.next());
    }
  }

  private final Shape shape;
}
