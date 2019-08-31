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
package org.tensorflow.nio.nd.impl.dimension;

import org.tensorflow.nio.nd.index.Index;

public final class Dimensions {

  public static Dimension unknown() {
    return UnknownDimension.INSTANCE;
  }

  public static Dimension axis(long numElements, long elementSize) {
    return new Axis(numElements, elementSize);
  }

  public static Dimension coord(long index, Dimension originalDimension) {
    return new Coordinate(index, (AbstractDimension) originalDimension);
  }

  public static Dimension indexed(Dimension originalDimension, Index index) {
    return new IndexedDimension((AbstractDimension) originalDimension, index);
  }
}
