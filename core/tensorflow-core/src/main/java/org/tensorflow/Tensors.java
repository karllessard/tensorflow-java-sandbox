/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import java.nio.charset.Charset;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TDouble;
import org.tensorflow.types.TFloat;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUInt8;

/** DataType-safe factory methods for creating {@link org.tensorflow.Tensor} objects. */
public final class Tensors {
  private Tensors() {}

  public static <T extends Tensor<?>> T of(DataType<T> dataType, Shape shape) {
    return Tensor.allocate(dataType, shape);
  }

  public static <T extends Tensor<U>, U> T copyOf(DataType<T> dataType, NdArray<U> data) {
    return Tensor.allocate(dataType, data);
  }

  public static <T extends Tensor<U>, U> T scalar(DataType<T> dataType, U value) {
    return Tensor.allocate(dataType, value);
  }

  public static TUInt8 scalar(byte value) {
    return scalar(TUInt8.DTYPE, value);
  }

  public static TInt32 scalar(int value) {
    return scalar(TInt32.DTYPE, value);
  }

  public static TInt64 scalar(long value) {
    return scalar(TInt64.DTYPE, value);
  }

  public static TFloat scalar(float value) {
    return scalar(TFloat.DTYPE, value);
  }

  public static TDouble scalar(double value) {
    return scalar(TDouble.DTYPE, value);
  }

  public static TString scalar(String value, Charset charset) {
    //return scalar(TString.DTYPE, value.getBytes(charset)); // TODO charset!
    return scalar(TString.DTYPE, value); // TODO charset!
  }

  public static TBool scalar(boolean value) {
    return scalar(TBool.DTYPE, value);
  }

  public static <T extends Tensor<U>, U> T vector(DataType<T> dataType, U[] values) {
    return Tensor.allocate(dataType, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TUInt8 vector(byte[] values) {
    return Tensor.allocate(TUInt8.DTYPE, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TInt32 vector(int[] values) {
    return Tensor.allocate(TInt32.DTYPE, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TInt64 vector(long[] values) {
    return Tensor.allocate(TInt64.DTYPE, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TFloat vector(float[] values) {
    return Tensor.allocate(TFloat.DTYPE, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TDouble vector(double[] values) {
    return Tensor.allocate(TDouble.DTYPE, NdArrays.wrap(values, Shape.make(values.length)));
  }

  public static TString vector(String[] values, Charset charset) {
    return Tensor.allocate(TString.DTYPE, NdArrays.wrap(values, Shape.make(values.length))); // TODO charset
  }

  public static TBool vector(boolean[] values) {
    return null; // TODO!
  }
}
