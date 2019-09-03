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

import java.nio.BufferOverflowException;
import java.nio.BufferUnderflowException;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.ValueIterable;
import org.tensorflow.nio.nd.ValueIterator;
import org.tensorflow.nio.nd.impl.iterator.Iterators;
import org.tensorflow.nio.nd.index.Index;

public class DefaultNdArray<T> implements NdArray<T> {

  public static <T> NdArray<T> wrap(DataBuffer<T> buffer, Shape shape) {
    Validator.denseShape(shape);
    return new DefaultNdArray<>(buffer, shape);
  }

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
  public Iterable<NdArray<T>> childElements() {
    return () -> Iterators.elementsOf(this);
  }

  @Override
  public NdArray<T> at(long... coordinates) {
    Shape sliceShape = shape().subshape(coordinates.length);
    long slicePosition = position(coordinates, false);
    return allocateSlice(slicePosition, sliceShape);
  }

  @Override
  public NdArray<T> slice(Index... indices) {
    Shape sliceShape = shape().mapTo(indices);
    long slicePosition = 0L;
    int i = 0;
    while (i < sliceShape.numDimensions() && sliceShape.dimension(i).numElements() == 0) {
      slicePosition += sliceShape.dimension(i++).position();
    }
    if (i > 0) {
      sliceShape = sliceShape.subshape(i);
    }
    return allocateSlice(slicePosition, sliceShape);
  }

  @Override
  public T get(long... coordinates) {
    return buffer.get(position(coordinates, true));
  }

  @Override
  public NdArray<T> set(T value, long... coordinates) {
    buffer.put(position(coordinates, true), value);
    return this;
  }

  @Override
  public NdArray<T> copyTo(NdArray<T> dst) {
    // TODO Optimize when array is continuous in memory
    slowCopy(this, dst);
    return this;
  }

  @Override
  public NdArray<T> copyFrom(NdArray<T> src) {
    // TODO Optimize when array is continuous in memory
    slowCopy(src, this);
    return this;
  }

  @Override
  public NdArray<T> read(T[] dst) {
    return read(DataBuffers.wrap(dst, false));
  }

  @Override
  public NdArray<T> read(T[] dst, int offset) {
    return read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public NdArray<T> read(DataBuffer<T> dst) {
    if (dst.remaining() < size()) {
      throw new BufferOverflowException();
    }
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.execute(this, (buffer, size) -> dst.put(buffer.limit(size)));
    } else {
      slowRead(dst);
    }
    return this;
  }

  @Override
  public NdArray<T> write(T[] src) {
    return write(DataBuffers.wrap(src, false));
  }

  @Override
  public NdArray<T> write(T[] src, int offset) {
    return write(DataBuffers.wrap(src, false).position(offset));
  }

  @Override
  public NdArray<T> write(DataBuffer<T> src) {
    if (src.remaining() < size()) {
      throw new BufferUnderflowException();
    }
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.execute(this, (buffer, size) -> buffer.put(src.limit(src.position() + size)));
    } else {
      slowWrite(src);
    }
    return this;
  }

  DataBuffer<T> buffer() {
    return buffer;
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

  private final DataBuffer<T> buffer;
  private final Shape shape;

  protected DefaultNdArray(DataBuffer<T> buffer, Shape shape) {
    this.buffer = buffer;
    this.shape = shape;
  }

  private DefaultNdArray<T> allocateSlice(long position, Shape shape) {
    return new DefaultNdArray<>(buffer.withPosition(position).slice(), shape);
  }

  private long position(long[] indices, boolean scalar) {
    if (indices.length > shape().numDimensions()) {
      throw new IndexOutOfBoundsException();
    }
    long position = 0L;
    int i = 0;
    for (; i < indices.length; ++i) {
      position += shape().dimension(i).positionOf(indices[i]);
    }
    while (i < shape().numDimensions() && shape().dimension(i).numElements() == 0) {
      position += shape().dimension(i++).position();
    }
    if (scalar && i < shape().numDimensions()) {
      throw new IllegalRankException("Not a scalar value");
    }
    return position;
  }

  /**
   * Check if we copy this array data in bulk. Bulk copy is only possible for array of 1-dimension
   * or more and that the last dimension is not segmented (therefore linear in memory).
   *
   * @return true if bulk copy is possible
   */
  private boolean isBulkCopyAvailable() {
    return shape().numDimensions() > 0 && !shape().dimension(shape().numDimensions() - 1)
        .isSegmented();
  }
}