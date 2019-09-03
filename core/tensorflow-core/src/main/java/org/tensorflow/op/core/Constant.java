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

package org.tensorflow.op.core;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.sun.org.apache.xpath.internal.operations.Bool;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.Charset;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Operator;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TDouble;
import org.tensorflow.types.TFloat;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;

/** An operator producing a constant value. */
@Operator
public final class Constant<T> extends PrimitiveOp implements Operand<T> {

  /**
   * Creates a constant containing a single {@code int} element.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The value to put into the new constant.
   * @return an integer constant
   */
  public static Constant<TInt32> create(Scope scope, int data) {
    try (TInt32 t = Tensors.scalar(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a rank-1 constant of {@code int} elements.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data An array containing the values to put into the new constant. The dimensions of the
   *     new constant will match those of the array.
   */
  public static Constant<TInt32> create(Scope scope, int[] data) {
    try (TInt32 t = Tensors.vector(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a constant containing a single {@code float} element.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The value to put into the new constant. 
   * @return a float constant
   */
  public static Constant<TFloat> create(Scope scope, float data) {
    try (TFloat t = Tensors.scalar(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a rank-1 constant of {@code float} elements.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data An array containing the values to put into the new constant. The dimensions of the
   *     new constant will match those of the array.
   */
  public static Constant<TFloat> create(Scope scope, float[] data) {
    try (TFloat t = Tensors.vector(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a constant containing a single {@code double} element.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The value to put into the new constant.
   * @return a double constant
   */
  public static Constant<TDouble> create(Scope scope, double data) {
    try (TDouble t = Tensors.scalar(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a rank-1 constant of {@code double} elements.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data An array containing the values to put into the new constant. The dimensions of the
   *     new constant will match those of the array.
   */
  public static Constant<TDouble> create(Scope scope, double[] data) {
    try (TDouble t = Tensors.vector(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a constant containing a single {@code long} element.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The value to put into the new constant.
   * @return a long constant
   */
  public static Constant<TInt64> create(Scope scope, long data) {
    try (TInt64 t = Tensors.scalar(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a rank-1 constant of {@code long} elements.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data An array containing the values to put into the new constant. The dimensions of the
   *     new constant will match those of the array.
   */
  public static Constant<TInt64> create(Scope scope, long[] data) {
    try (TInt64 t = Tensors.vector(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a constant containing a single {@code boolean} element.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The value to put into the new constant.
   * @return a boolean constant
   */
  public static Constant<TBool> create(Scope scope, boolean data) {
    try (TBool t = Tensors.scalar(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a rank-1 constant of {@code boolean} elements.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data An array containing the values to put into the new constant. The dimensions of the
   *     new constant will match those of the array.
   */
  public static Constant<TBool> create(Scope scope, boolean[] data) {
    try (TBool t = Tensors.vector(data)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a {@code String} constant using the default, UTF-8 encoding.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param data The string to put into the new constant.
   * @return a string constant
   */
  public static Constant<TString> create(Scope scope, String data) {
    try (TString t = Tensors.scalar(data, UTF_8)) {
      return create(scope, t);
    }
  }

  /**
   * Creates a {@code String} constant using a specified encoding.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param charset The encoding from String to bytes.
   * @param data The string to put into the new constant.
   * @return a string constant
   */
  public static Constant<TString> create(Scope scope, String data, Charset charset) {
    try (TString t = Tensors.scalar(data, charset)) {
      return create(scope, t);
    }
  }

  /**
   * Create a constant with data from the given buffer.
   *
   * <p>Creates a Constant with the provided shape of any type where the constant data has been
   * encoded into {@code data} as per the specification of the TensorFlow <a
   * href="https://www.tensorflow.org/code/tensorflow/c/c_api.h">C
   * API</a>.
   *
   * @param scope is a scope used to add the underlying operation.
   * @param type the tensor datatype.
   * @param shape the tensor shape.
   * @param data a buffer containing the tensor data.
   * @return a constant of type `type`
   * @throws IllegalArgumentException If the tensor datatype or shape is not compatible with the
   *     buffer
   */
  public static <T extends Tensor<U>, U> Constant<T> create(Scope scope, DataType<T> dataType,
      NdArray<U> data) {
    try (T t = Tensors.copyOf(dataType, data)) {
      return create(scope, t);
    }
  }

  /**
   * Create a constant from a Java object.
   *
   * <p>The argument {@code object} is first converted into a Tensor using {@link
   * org.tensorflow.Tensor#create(Object)}, so only Objects supported by this method must be
   * provided. For example:
   *
   * <pre>{@code
   * Constant.create(scope, new int[]{{1, 2}, {3, 4}}, Integer.class); // returns a 2x2 integer matrix
   * }</pre>
   *
   * @param scope is a scope used to add the underlying operation.
   * @param object a Java object representing the constant.
   * @return a constant of type `type`
   * @see org.tensorflow.Tensor#create(Object) Tensor.create
   */
  public static <T extends Tensor<?>> Constant<T> create(Scope scope, T tensor) {
    return new Constant<>(
        scope
            .env()
            .opBuilder("Const", scope.makeOpName("Const"))
            .setAttr("value", tensor)
            .setAttr("dtype", tensor.dataType())
            .build());
  }

  @Override
  public Output<T> asOutput() {
    return output;
  }

  private Constant(Operation operation) {
    super(operation);
    output = operation.output(0);
  }

  private final Output<T> output;
}
