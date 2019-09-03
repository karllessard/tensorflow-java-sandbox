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
package org.tensorflow;

import java.nio.ByteBuffer;
import org.tensorflow.nio.nd.Shape;

public final class DataType<T> {

  @FunctionalInterface
  public interface Instantiator<T> {
    T instantiate(Shape shape, long handle, ByteBuffer rawBuffer);
  }

  // Declare T to extends Tensor<?> so we know for sure that all objects instantiated from
  // a datatype is a tensor (we won't carry that restriction in the DataType parameter for
  // readability)
  public static <T extends Tensor<?>> DataType<T> make(int ordinal, int byteSize,
      Instantiator<T> instantiator) {
    return new DataType<>(ordinal, byteSize, instantiator);
  }

  public int byteSize() {
    return byteSize;
  }

  public int ordinal() {
    return ordinal;
  }

  Instantiator<T> instantiator() {
    return instantiator;
  }

  boolean isVariableLength() {
    return byteSize == -1;
  }

  private final int ordinal;
  private final int byteSize;
  private final Instantiator<T> instantiator;

  private DataType(int ordinal, int byteSize, Instantiator<T> instantiator) {
    this.ordinal = ordinal;
    this.byteSize = byteSize;
    this.instantiator = instantiator;
  }
}
