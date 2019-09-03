/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
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

@RunWith(JUnit4.class)
public class ConstantTest {

  @Test
  public void createInt32() {
    Integer value = 1;

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<TInt32> op = Constant.create(scope, value);
      try (TInt32 result = sess.runner().fetch(op).run().get(0).expect(TInt32.DTYPE)) {
        assertEquals(value, result.get());
      }
    }
  }

  @Test
  public void createFloat() {
    Float value = 1.0f;

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<TFloat> op = Constant.create(scope, value);
      try (TFloat result = sess.runner().fetch(op).run().get(0).expect(TFloat.DTYPE)) {
        assertEquals(value, result.get(), 0.0f);
      }
    }
  }

  @Test
  public void createDouble() {
    Double value = 1.0;

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<TDouble> op = Constant.create(scope, value);
      try (TDouble result = sess.runner().fetch(op).run().get(0).expect(TDouble.DTYPE)) {
        assertEquals(value, result.get(), 0.0);
      }
    }
  }

  @Test
  public void createInt64() {
    Long value = 1L;

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<TInt64> op = Constant.create(scope, value);
      try (TInt64 result = sess.runner().fetch(op).run().get(0).expect(TInt64.DTYPE)) {
        assertEquals(value, result.get());
      }
    }
  }

  @Test
  public void createBool() {
    Boolean value = true;

    try (Graph g = new Graph();
        Session sess = new Session(g)) {
      Scope scope = new Scope(g);
      Constant<TBool> op = Constant.create(scope, value);
      try (TBool result = sess.runner().fetch(op).run().get(0).expect(TBool.DTYPE)) {
        assertEquals(value, result.get());
      }
    }
  }
}
