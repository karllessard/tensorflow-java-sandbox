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

import org.tensorflow.TensorFlow;

public abstract class DataType<T> {

  public Class<T> javaType() {
    return javaType;
  }

  public int ordinal() {
    return ordinal;
  }

  public int byteSize() {
    return byteSize;
  }

  DataType(Class<T> javaType, int ordinal, int byteSize) {
    this.javaType = javaType;
    this.ordinal = ordinal;
    this.byteSize = byteSize;
  }

  private Class<T> javaType;
  private int ordinal;
  private int byteSize;

  public static DataType<?> valueOf(int ordinal) {
    switch (ordinal) {
      case TFloat.ORDINAL: return TFloat.TYPE;
      case TDouble.ORDINAL: return TDouble.TYPE;
      case Int32.ORDINAL: return Int32.TYPE;
      case Int64.ORDINAL: return Int64.TYPE;
      case TString.ORDINAL: return TString.TYPE;
      case UInt8.ORDINAL: return UInt8.TYPE;
      case Bool.ORDINAL: return Bool.TYPE;
      default:
        throw new IllegalArgumentException("DataType " + ordinal +
            " is not recognized in Java (version " + TensorFlow.version() + ")");
    }
  }
}
