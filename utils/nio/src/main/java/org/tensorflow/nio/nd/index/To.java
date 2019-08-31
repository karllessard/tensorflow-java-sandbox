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

import org.tensorflow.nio.nd.impl.dimension.Dimension;

/**
 * An index that returns only elements on a given dimension up to a specific coordinate.
 *
 * <p>For example, given a vector with {@code n} elements on the {@code x} axis, and {@code n > k},
 * {@code to(k)} returns x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>k</sub>
 */
class To implements Index {

  @Override
  public long numElements(Dimension dim) {
    return end;
  }

  @Override
  public long mapCoordinate(long coordinate, Dimension dim) {
    return coordinate;
  }

  @Override
  public Dimension apply(Dimension dim) {
    if (end > dim.numElements()) {
      throw new IndexOutOfBoundsException("End coordinate exceeds the number of elements");
    }
    return end == dim.numElements() ? dim : Index.super.apply(dim);
  }

  To(long end) {
    this.end = end;
  }

  private long end;
}
