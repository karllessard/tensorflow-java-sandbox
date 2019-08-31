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
package org.tensorflow.nio.nd.index;

import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.impl.dimension.Dimension;
import org.tensorflow.nio.nd.impl.dimension.Dimensions;

/**
 * An index that returns only specific elements on a given dimension.
 *
 * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n >
 * 10}, {@code seq(8, 0, 3)} returns x<sub>8</sub>, x<sub>0</sub>, x<sub>3</sub>
 */
class Sequence implements Index {

  @Override
  public long numElements(Dimension dim) {
    return values.size();
  }

  @Override
  public long mapCoordinate(long coordinate, Dimension dim) {
    return values.get(coordinate).longValue();
  }

  Sequence(NdArray<? extends Number> values) {
    this.values = values;
  }

  private NdArray<? extends Number> values;
}
