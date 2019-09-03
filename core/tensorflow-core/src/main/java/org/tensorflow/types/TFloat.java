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
package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.types.family.Decimal;

public class TFloat extends Tensor<Float> implements Decimal {

  public static final DataType<TFloat> DTYPE = DataType.make(1, 4, TFloat::new);

  private TFloat(Shape shape, long handle, ByteBuffer rawBuffer) {
    super(DTYPE, shape, handle, DataBuffers.wrap(rawBuffer.asFloatBuffer()));
  }
}
