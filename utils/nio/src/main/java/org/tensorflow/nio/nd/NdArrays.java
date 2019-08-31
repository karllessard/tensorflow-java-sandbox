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

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.impl.dense.ByteDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.DenseNdArray;
import org.tensorflow.nio.nd.impl.dense.DoubleDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.FloatDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.IntDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.LongDenseNdArray;

/**
 * Helper class for creating {@link NdArray} instances
 */
public final class NdArrays {

  /**
   * Creates an N-dimensional array of bytes of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static ByteNdArray ofBytes(Shape shape) {
    return wrap(DataBuffers.ofBytes(shape.size()), shape);
  }

  /**
   * Wraps a byte array into an N-dimensional array
   *
   * @param values byte array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static ByteNdArray wrap(byte[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps a byte data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static ByteNdArray wrap(ByteDataBuffer buffer, Shape shape) {
    return ByteDenseNdArray.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of longs of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static LongNdArray ofLongs(Shape shape) {
    return wrap(DataBuffers.ofLongs(shape.size()), shape);
  }

  /**
   * Wraps a long array into an N-dimensional array
   *
   * @param values integer array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static LongNdArray wrap(long[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps a long data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static LongNdArray wrap(LongDataBuffer buffer, Shape shape) {
    return LongDenseNdArray.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of integers of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static IntNdArray ofIntegers(Shape shape) {
    return wrap(DataBuffers.ofIntegers(shape.size()), shape);
  }

  /**
   * Wraps an integer array into an N-dimensional array
   *
   * @param values integer array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static IntNdArray wrap(int[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps an integer data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static IntNdArray wrap(IntDataBuffer buffer, Shape shape) {
    return IntDenseNdArray.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of floats of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static FloatNdArray ofFloats(Shape shape) {
    return wrap(DataBuffers.ofFloats(shape.size()), shape);
  }

  /**
   * Wraps a float array into an N-dimensional array
   *
   * @param values float array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static FloatNdArray wrap(float[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps a float data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static FloatNdArray wrap(FloatDataBuffer buffer, Shape shape) {
    return FloatDenseNdArray.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of doubles of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static DoubleNdArray ofDoubles(Shape shape) {
    return wrap(DataBuffers.ofDoubles(shape.size()), shape);
  }

  /**
   * Wraps a double array into an N-dimensional array
   *
   * @param values double array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static DoubleNdArray wrap(double[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps a double data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static DoubleNdArray wrap(DoubleDataBuffer buffer, Shape shape) {
    return DoubleDenseNdArray.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of objects of the given shape
   *
   * @param clazz type of the objects to be stored in the array
   * @param shape shape of the array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> of(Class<T> clazz, Shape shape) {
    return wrap(DataBuffers.of(clazz, shape.size()), shape);
  }

  /**
   * Wraps an object array into an N-dimensional array
   *
   * @param values object array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> wrap(T[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Wraps a data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> wrap(DataBuffer<T> buffer, Shape shape) {
    return DenseNdArray.wrap(buffer, shape);
  }
}

