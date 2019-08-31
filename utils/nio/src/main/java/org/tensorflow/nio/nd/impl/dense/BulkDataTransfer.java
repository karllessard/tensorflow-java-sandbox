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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.nd.impl.dimension.Dimension;

final class BulkDataTransfer {

  @FunctionalInterface
  interface BulkCopy<T> {

    void invoke(DataBuffer<T> arrayBuffer, long bulkCopySize);
  }

  /**
   * Copy in bulk the given array by invoking recursively the {@code bulkCopy} operation
   * for each chunk of contiguous data.
   *
   * @param array array implied in the copy
   * @param bulkCopy the copy operation that should be invoked for each chunk of contiguous data
   * @param <T> type of data
   */
  static <T> void execute(AbstractDenseNdArray<T, ?> array, BulkCopy<T> bulkCopy) {
    int bulkCopyDimensionIdx = -1;
    long bulkCopySize = 1L;

    // Find what are the biggest chunk of data that we can copy in bulk by starting from the
    // last dimension of this array and iterating backward until we hit a dimension that is
    // segmented (if any)
    for (int i = array.shape().numDimensions() - 1; i >= 0; --i) {
      Dimension dim = array.shape().dimension(i);
      if (dim.isSegmented()) {
        break;
      }
      bulkCopyDimensionIdx = i;
      bulkCopySize *= dim.numElements();
    }
    if (bulkCopyDimensionIdx < 0) {
      throw new IllegalArgumentException(
          "This array cannot be copied in bulk, since its last dimension is segmented");
    }
    copyRecursively(bulkCopy, bulkCopyDimensionIdx, bulkCopySize, array, 0);
  }

  /**
   * Recursively copy the data in bulk of the given element.
   *
   * @param bulkCopy the bulk copy operation
   * @param bulkCopyDimensionIdx index of the first dimension that can be copied in bulk
   * @param bulkCopySize number of values that can be copied in a single bulk operation
   * @param currentElement the current element
   * @param currentDimensionIdx the index of the dimension of the current element
   * @param <T> type of data
   */
  private static <T> void copyRecursively(
      BulkCopy<T> bulkCopy,
      int bulkCopyDimensionIdx,
      long bulkCopySize,
      AbstractDenseNdArray<T, ?> currentElement,
      int currentDimensionIdx
  ) {
    if (currentDimensionIdx == bulkCopyDimensionIdx) {
      bulkCopy.invoke(currentElement.buffer().duplicate(), bulkCopySize);
    } else {
      currentElement.childElements().forEach(e -> copyRecursively(
          bulkCopy,
          bulkCopyDimensionIdx,
          bulkCopySize,
          (AbstractDenseNdArray<T, ?>) e,
          currentDimensionIdx + 1)
      );
    }
  }
}
