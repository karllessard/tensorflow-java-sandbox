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
package org.tensorflow.nio.nd.impl.dense;

import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.Shape;

public class FloatDenseNdArray extends AbstractDenseNdArray<Float, FloatNdArray> implements
    FloatNdArray {

  public static FloatNdArray wrap(FloatDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new FloatDenseNdArray(buffer, shape);
  }

  @Override
  protected FloatDataBuffer buffer() {
    return buffer;
  }

  protected FloatDenseNdArray allocateSlice(long position, Shape shape) {
    return new FloatDenseNdArray(buffer.withPosition(position).slice(), shape);
  }

  private FloatDenseNdArray(FloatDataBuffer buffer, Shape shape) {
    super(shape);
    this.buffer = buffer;
  }

  private FloatDataBuffer buffer;
}