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
package org.tensorflow.nio;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;
import org.tensorflow.nio.nd.index.Indices;

/**
 * A static API for interfacing with the NIO library.
 */
public class StaticApi {

  /**
   * Creates a buffer of objects of type `clazz` that can store up to `capacity` values
   *
   * @param clazz the type of object stored in this buffer
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static <T> DataBuffer<T> bufferOf(Class<T> clazz, long capacity) {
    return DataBuffers.of(clazz, capacity);
  }

  /**
   * Wraps an array of objects into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static <T> DataBuffer<T> bufferOf(T[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates a buffer of bytes that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static ByteDataBuffer bufferOfBytes(long capacity) {
    return DataBuffers.ofBytes(capacity);
  }

  /**
   * Wraps an array of bytes into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static ByteDataBuffer bufferOf(byte[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates a buffer of integers that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static IntDataBuffer bufferOfInts(long capacity) {
    return DataBuffers.ofIntegers(capacity);
  }

  /**
   * Wraps an array of integers into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static IntDataBuffer bufferOf(int[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates a buffer of longs that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static LongDataBuffer bufferOfLongs(long capacity) {
    return DataBuffers.ofLongs(capacity);
  }

  /**
   * Wraps an array of longs into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static LongDataBuffer bufferOf(long[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates a buffer of floats that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static FloatDataBuffer bufferOfFloats(long capacity) {
    return DataBuffers.ofFloats(capacity);
  }

  /**
   * Wraps an array of floats into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static FloatDataBuffer bufferOf(float[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates a buffer of doubles that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static DoubleDataBuffer bufferOfDoubles(long capacity) {
    return DataBuffers.ofDoubles(capacity);
  }

  /**
   * Wraps an array of doubles into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static DoubleDataBuffer bufferOf(double[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  /**
   * Creates an N-dimensional array of objects of the given shape
   *
   * @param clazz type of the objects to be stored in the array
   * @param shape shape of the array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> ndArrayOf(Class<T> clazz, Shape shape) {
    return NdArrays.of(clazz, shape);
  }

  /**
   * Wraps an object array into an N-dimensional array
   *
   * @param values object array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> ndArrayOf(T[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Wraps a data buffer into an N-dimensional array
   *
   * @param buffer buffer to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static <T> NdArray<T> ndArrayOf(DataBuffer<T> buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  /**
   * Creates an N-dimensional array of bytes of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Byte> ndArrayOfBytes(Shape shape) {
    return NdArrays.ofBytes(shape);
  }

  /**
   * Wraps a byte array into an N-dimensional array
   *
   * @param values byte array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Byte> ndArrayOf(byte[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Creates an N-dimensional array of integers of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Integer> ndArrayOfInts(Shape shape) {
    return NdArrays.ofInts(shape);
  }

  /**
   * Wraps an integer array into an N-dimensional array
   *
   * @param values integer array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Integer> ndArrayOf(int[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Creates an N-dimensional array of longs of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Long> ndArrayOfLongs(Shape shape) {
    return NdArrays.ofLongs(shape);
  }

  /**
   * Wraps a long array into an N-dimensional array
   *
   * @param values integer array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Long> ndArrayOf(long[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Creates an N-dimensional array of floats of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Float> ndArrayOfFloats(Shape shape) {
    return NdArrays.ofFloats(shape);
  }

  /**
   * Wraps a float array into an N-dimensional array
   *
   * @param values float array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Float> ndArrayOf(float[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Creates an N-dimensional array of doubles of the given shape
   *
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Double> ndArrayOfDoubles(Shape shape) {
    return NdArrays.ofDoubles(shape);
  }

  /**
   * Wraps a double array into an N-dimensional array
   *
   * @param values double array to wrap
   * @param shape shape of the N-dimensional array
   * @return the new N-dimensional array
   */
  public static NdArray<Double> ndArrayOf(double[] values, Shape shape) {
    return NdArrays.wrap(values, shape);
  }

  /**
   * Create a Shape representing an N-dimensional value.
   *
   * <p>Creates a Shape representing an N-dimensional value (N being at least 1), with the provided
   * size for each dimension. A -1 indicates that the size of the corresponding dimension is
   * unknown.
   *
   * @see Shape#make(long...)
   */
  public static Shape shape(long... dimensionSize) {
    return Shape.make(dimensionSize);
  }

  /**
   * An index that selects a specific element on a given dimension.
   *
   * <p>When this index is applied to a given dimension, the dimension is resolved as a
   * single element and therefore is excluded from the computation of the rank.
   *
   * <p>For example, given a 3D matrix on the axis [x, y, z], if
   * {@code matrix.slice(all(), at(0), at(0)}, then the rank of the returned slice is 1 and its
   * number of elements is {@code x.numElements()}
   *
   * @param coordinate coordinate of the element referenced by this index
   * @return an index
   */
  public static Index at(long coordinate) {
    return Indices.at(coordinate);
  }

  /**
   * An index that selects a specific element on a given dimension, as provided by a scalar
   * coordinate.
   *
   * <p>This is the equivalent of calling {@code at(scalar.get().longValue())}
   *
   * @param scalar scalar providing the coordinate of the element referenced by this index
   * @return an index
   * @see #at(long)
   */
  public static Index at(NdArray<? extends Number> scalar) {
    return Indices.at(scalar);
  }

  /**
   * An index that returns all elements of a dimension in the original order.
   *
   * <p>Applying this index to a given dimension will return the original dimension
   * directly.
   *
   * <p>For example, given a vector with {@code n} elements, {@code all()} returns
   * x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>n-1</sub>
   *
   * @return an index
   */
  public static Index all() {
    return Indices.all();
  }

  /**
   * An index that returns only specific elements on a given dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n >
   * 10}, {@code seq(8, 0, 3)} returns x<sub>8</sub>, x<sub>0</sub>, x<sub>3</sub>
   *
   * @param coordinates coordinates of the element referenced by this index, in the sequence order
   * @return an index
   */
  public static Index seq(long... coordinates) {
    return Indices.seq(coordinates);
  }

  /**
   * An index that returns only specific elements on a given dimension, as provided by a vector of
   * coordinates.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n >
   * 10}, {@code seq(8, 0, 3)} returns x<sub>8</sub>, x<sub>0</sub>, x<sub>3</sub>
   *
   * @param vector vector of coordinates of the element referenced by this index, in the sequence
   *               order
   * @return an index
   */
  public static Index seq(NdArray<? extends Number> vector) {
    return Indices.seq(vector);
  }

  /**
   * An index that returns only elements found at an even position in the original dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and n is even,
   * {@code even()} returns x<sub>0</sub>, x<sub>2</sub>, ..., x<sub>n-2</sub>
   *
   * @return an index
   */
  public static Index even() {
    return Indices.even();
  }

  /**
   * An index that returns only elements found at an odd position in the original dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and n is even,
   * {@code odd()} returns x<sub>1</sub>, x<sub>3</sub>, ..., x<sub>n-1</sub>
   *
   * @return an index
   */
  public static Index odd() {
    return Indices.odd();
  }

  /**
   * An index that returns only elements on a given dimension starting at a specific coordinate.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k}, {@code from(k)} returns x<sub>k</sub>, x<sub>k+1</sub>, ..., x<sub>n-1</sub>
   *
   * @param start coordinate of the first element referenced by this index
   * @return an index
   */
  public static Index from(long start) {
    return Indices.from(start);
  }

  /**
   * An index that returns only elements on a given dimension up to a specific coordinate.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k}, {@code to(k)} returns x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>k</sub>
   *
   * @param end coordinate of the last element referenced by this index
   * @return an index
   */
  public static Index to(long end) {
    return Indices.to(end);
  }

  /**
   * An index that returns only elements on a given dimension between two coordinates.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and
   * {@code n > k > j}, {@code range(j, k)} returns x<sub>j</sub>, x<sub>j+1</sub>, ...,
   * x<sub>k</sub>
   *
   * @param start coordinate of the first element referenced by this index
   * @param end coordinate of the last element referenced by this index
   * @return an index
   */
  public static Index range(long start, long end) {
    return Indices.range(start, end);
  }

  /**
   * An index that reverse the order of the elements on a given dimension.
   *
   * <p>For example, given a vector with {@code n} elements on the {@code x} axis,
   * this index returns x<sub>n-1</sub>, x<sub>n-2</sub>, ..., x<sub>0</sub>
   *
   * @return an index
   */
  public static Index flip() {
    return Indices.flip();
  }
}
