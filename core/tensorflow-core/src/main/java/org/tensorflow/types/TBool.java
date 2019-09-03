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
import org.tensorflow.memory.BooleanTensorBuffer;
import org.tensorflow.nio.nd.Shape;

public class TBool extends Tensor<Boolean> {

  public static final DataType<TBool> DTYPE = DataType.make(10, 1, TBool::new);

  private TBool(Shape shape, long handle, ByteBuffer rawBuffer) {
    super(DTYPE, shape, handle, new BooleanTensorBuffer(rawBuffer));
  }
}
