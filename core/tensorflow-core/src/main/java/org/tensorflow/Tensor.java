/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

import java.nio.ByteBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.DefaultNdArray;

/**
 * A statically typed multi-dimensional array whose elements are of a type described by T.
 *
 * <p>Instances of a Tensor are <b>not</b> thread-safe.
 *
 * <p><b>WARNING:</b> Resources consumed by the Tensor object <b>must</b> be explicitly freed by
 * invoking the {@link #close()} method when the object is no longer needed. For example, using a
 * try-with-resources block:
 *
 * <pre>{@code
 * try (Tensor t = Tensor.create(...)) {
 *   doSomethingWith(t);
 * }
 * }</pre>
 */
public class Tensor<T> extends DefaultNdArray<T> implements AutoCloseable {

  /**
   * Release resources associated with the Tensor.
   *
   * <p><b>WARNING:</b>This must be invoked for all tensors that were not been produced by an eager
   * operation or memory will be leaked.
   *
   * <p>The Tensor object is no longer usable after {@code close} returns.
   */
  @Override
  public void close() {
    nativeRef.release();
  }

  /** Returns the {@link DataType} of elements stored in the Tensor. */
  public DataType dataType() {
    return dtype;
  }

  /**
   * Returns this Tensor object with the type {@code Tensor<U>}. This method is useful when given a
   * value of type {@code Tensor<?>}.
   *
   * @param dataType any (non-null) array of the correct type.
   * @throws IllegalArgumentException if the actual data type of this object does not match the type
   *     {@code U}.
   */
  @SuppressWarnings("unchecked")
  public <U extends Tensor<?>> U expect(DataType<U> dataType) {
    if (!dataType.equals(dtype)) {
      throw new IllegalArgumentException(
          "Cannot cast from tensor of " + dtype + " to tensor of " + dataType);
    }
    return (U) this;
  }

  static <T extends Tensor<?>> T allocate(DataType<T> dtype, Shape shape) {
    if (dtype.isVariableLength()) {
      throw new IllegalArgumentException("Only tensor of fixed-length data types can be allocated"
          + " without data");
    }
    long handle = allocate(dtype.ordinal(), shape.toArray(), dtype.byteSize());
    ByteBuffer buffer = buffer(handle);
    return dtype.instantiator().instantiate(shape, handle, buffer);
  }

  static <T extends Tensor<U>, U> T allocate(DataType<T> dtype, U value) {
    if (dtype.isVariableLength()) {
      throw new UnsupportedOperationException(); // TODO!
    }
    long handle = allocate(dtype.ordinal(), new long[] {}, dtype.byteSize());
    ByteBuffer buffer = buffer(handle);
    T tensor = dtype.instantiator().instantiate(Shape.make(), handle, buffer);
    tensor.set(value);
    return tensor;
  }

  static <T extends Tensor<U>, U> T allocate(DataType<T> dataType, NdArray<U> data) {
    if (dataType.isVariableLength()) {
      throw new UnsupportedOperationException(); // TODO!
    }
    long handle = allocate(dataType.ordinal(), data.shape().toArray(), dataType.byteSize());
    ByteBuffer buffer = buffer(handle);
    T tensor = dataType.instantiator().instantiate(data.shape(), handle, buffer);
    tensor.copyFrom(data);
    return tensor;
  }

  /**
   * Create a Tensor object from a handle to the C TF_Tensor object.
   *
   * <p>Takes ownership of the handle.
   */
  static Tensor<?> fromHandle(long handle) {
    DataType<?> dtype = DataTypes.fromOrdinal(dtype(handle));
    Shape shape = Shape.make(shape(handle));
    ByteBuffer buffer = buffer(handle);
    // It is ok to cast the instantiated object to a Tensor<?> since only types extending it
    // can be used to create a DataType
    return (Tensor<?>) dtype.instantiator().instantiate(shape, handle, buffer);
  }

  /**
   * Create an eager Tensor object from a handle to the C TF_Tensor object.
   *
   * <p>Takes ownership of the handle.
   */
  static Tensor<?> fromHandle(long handle, EagerSession session) {
    Tensor<?> t = fromHandle(handle);
    t.nativeRef.eager(session, t);
    return t;
  }

  long getNativeHandle() {
    return nativeRef.tensorHandle;
  }

  protected Tensor(DataType dtype, Shape shape, long handle, DataBuffer<T> buffer) {
    super(buffer, shape);
    this.dtype = dtype;
    this.nativeRef = new NativeReference(handle);
  }

  private final DataType dtype;
  private NativeReference nativeRef = null;

  /**
   * Reference to the underlying native tensor
   *
   * <p>Tensors are commonly allocated in a `try-with-resources` statement, where they get
   * automatically released after executing the last line of the `try` block they were declared in.
   *
   * <p>They can also be attached to an eager session, where in this case their lifetime ends either
   * when this session is closed or when the Tensor instance is no longer referenced and have been
   * garbage-collected.
   *
   * <p>This helper class wraps the tensor native handle and support both situations; If an eager
   * reference to the tensor exists, it will take care of releasing the tensor at the end of its
   * life. If the tensor is being explicitly closed before this happens, it will take cake of
   * clearing its association with any eager session before cleaning up the resources.
   */
  private static class NativeReference {

    /** Attaches this reference to an eager session */
    private class EagerReference extends EagerSession.NativeReference {

      EagerReference(EagerSession session, Tensor<?> tensor) {
        super(session, tensor);
      }

      @Override
      void delete() {
        // Mark this eager reference as cleared since it has been deleted by the session
        NativeReference.this.eagerRef = null;
        NativeReference.this.release();
      }
    }

    NativeReference(long tensorHandle) {
      this.tensorHandle = tensorHandle;
    }

    void eager(EagerSession session, Tensor<?> tensor) {
      if (eagerRef != null) {
        throw new IllegalStateException("The tensor is already attached to an eager session");
      }
      eagerRef = new EagerReference(session, tensor);
    }

    synchronized void release() {
      if (tensorHandle != 0L) {
        // Clear any remaining eager reference to this tensor
        if (eagerRef != null) {
          eagerRef.clear();
          eagerRef = null;
        }
        Tensor.delete(tensorHandle);
        tensorHandle = 0L;
      }
    }

    private long tensorHandle;
    private EagerReference eagerRef;
  }

  private static native long allocate(int dtype, long[] shape, long byteSize);

  private static native long allocateScalarBytes(byte[] value);

  private static native long allocateNonScalarBytes(long[] shape, Object[] value);

  private static native void delete(long handle);

  private static native ByteBuffer buffer(long handle);

  private static native int dtype(long handle);

  private static native long[] shape(long handle);

  private static native void setValue(long handle, Object value);

  private static native float scalarFloat(long handle);

  private static native double scalarDouble(long handle);

  private static native int scalarInt(long handle);

  private static native long scalarLong(long handle);

  private static native boolean scalarBoolean(long handle);

  private static native byte[] scalarBytes(long handle);

  private static native void readNDArray(long handle, Object value);

  static {
    TensorFlow.init();
  }
}
