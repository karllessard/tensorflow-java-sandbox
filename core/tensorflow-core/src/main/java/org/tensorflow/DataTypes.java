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

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TDouble;
import org.tensorflow.types.TFloat;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUInt8;

class DataTypes {

  static {
    register(
        TFloat.DTYPE,
        TDouble.DTYPE,
        TInt32.DTYPE,
        TInt64.DTYPE,
        TString.DTYPE,
        TUInt8.DTYPE,
        TBool.DTYPE
    );
  }

  static DataType fromOrdinal(int ordinal) {
    DataType<?> dataType = dataTypes.get(ordinal);
    if (dataType == null) {
      throw new IllegalArgumentException("DataType " + ordinal +
          " is not recognized in Java (version " + TensorFlow.version() + ")");
    }
    return dataType;
  }

  private static Map<Integer, DataType<?>> dataTypes;

  private static void register(DataType<?>... dtypes) {
    dataTypes = Arrays.stream(dtypes)
        .collect(Collectors.toMap(DataType::ordinal, Function.identity()));
  }
}
