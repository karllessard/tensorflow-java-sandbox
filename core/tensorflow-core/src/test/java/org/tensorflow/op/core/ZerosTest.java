/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.op.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TDouble;
import org.tensorflow.types.TFloat;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUInt8;

@RunWith(JUnit4.class)
public class ZerosTest {

  @Test
  public void createInt32Zeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TInt32> op = Zeros.create(scope, Constant.create(scope, shape), TInt32.DTYPE);
      try (TInt32 result = sess.runner().fetch(op).run().get(0).expect(TInt32.DTYPE)) {
        result.values().forEach(v -> assertEquals(Integer.valueOf(0), v));
      }
    }
  }

  @Test
  public void createFloatZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TFloat> op = Zeros.create(scope, Constant.create(scope, shape), TFloat.DTYPE);
      try (TFloat result = sess.runner().fetch(op).run().get(0).expect(TFloat.DTYPE)) {
        result.values().forEach(v -> assertEquals(Float.valueOf(0), v));
      }
    }
  }

  @Test
  public void createDoubleZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TDouble> op = Zeros.create(scope, Constant.create(scope, shape), TDouble.DTYPE);
      try (TDouble result = sess.runner().fetch(op.asOutput()).run().get(0).expect(TDouble.DTYPE)) {
        result.values().forEach(v -> assertEquals(Double.valueOf(0), v));
      }
    }
  }

  @Test
  public void createInt64Zeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TInt64> op = Zeros.create(scope, Constant.create(scope, shape), TInt64.DTYPE);
      try (TInt64 result = sess.runner().fetch(op.asOutput()).run().get(0).expect(TInt64.DTYPE)) {
        result.values().forEach(v -> assertEquals(Long.valueOf(0), v));
      }
    }
  }

  @Test
  public void createBoolZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TBool> op = Zeros.create(scope, Constant.create(scope, shape), TBool.DTYPE);
      try (TBool result = sess.runner().fetch(op.asOutput()).run().get(0).expect(TBool.DTYPE)) {
        result.values().forEach(Assert::assertFalse);
      }
    }
  }

  @Test
  public void createUInt8Zeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros<TUInt8> op = Zeros.create(scope, Constant.create(scope, shape), TUInt8.DTYPE);
      try (TUInt8 result = sess.runner().fetch(op.asOutput()).run().get(0).expect(TUInt8.DTYPE)) {
        result.values().forEach(v -> assertEquals(Byte.valueOf((byte)0), v));
      }
    }
  }
  
  @Test(expected = IllegalArgumentException.class)
  public void cannotCreateStringZeros() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros.create(scope, Constant.create(scope, shape), TString.DTYPE);
    }
  }
  
  @Test
  public void operationsComposingZerosAreCorrectlyNamed() {
    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      long[] shape = {2, 2};
      Zeros.create(scope.withSubScope("test"), Constant.create(scope, shape), TFloat.DTYPE);
      sess.runner().addTarget("test/Zeros/Zero").addTarget("test/Zeros/Fill").run();
    }
  }
}
