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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.impl.DefaultNdArray;

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
  public static NdArray<Byte> ofBytes(Shape shape) {
    return wrap(DataBuffers.ofBytes(shape.size()), shape);
  }

  /**
   * Wraps a byte array into an N-dimensional array
   *
   * @param values byte array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Byte> wrap(byte[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Creates an N-dimensional array of longs of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Long> ofLongs(Shape shape) {
    return wrap(DataBuffers.ofLongs(shape.size()), shape);
  }

  /**
   * Wraps a long array into an N-dimensional array
   *
   * @param values long array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Long> wrap(long[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Creates an N-dimensional array of integers of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Integer> ofInts(Shape shape) {
    return wrap(DataBuffers.ofIntegers(shape.size()), shape);
  }

  /**
   * Wraps an integer array into an N-dimensional array
   *
   * @param values integer array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Integer> wrap(int[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Creates an N-dimensional array of floats of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Float> ofFloats(Shape shape) {
    return wrap(DataBuffers.ofFloats(shape.size()), shape);
  }
  /**
   * Wraps a float array into an N-dimensional array
   *
   * @param values float array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Float> wrap(float[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
  }

  /**
   * Creates an N-dimensional array of doubles of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Double> ofDoubles(Shape shape) {
    return wrap(DataBuffers.ofDoubles(shape.size()), shape);
  }

  /**
   * Wraps a double array into an N-dimensional array
   *
   * @param values double array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Double> wrap(double[] values, Shape shape) {
    return wrap(DataBuffers.wrap(values, false), shape);
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
    return DefaultNdArray.wrap(buffer, shape);
  }
}

