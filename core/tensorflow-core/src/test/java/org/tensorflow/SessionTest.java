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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.types.TInt32;

/** Unit tests for {@link Session}. */
@RunWith(JUnit4.class)
public class SessionTest {

  @Test
  public void runUsingOperationNames() {
    try (Graph g = new Graph();
        Session s = new Session(g);
        TInt32 a = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
      a.set(2, 0, 0);
      a.set(3, 0, 0);
      TestUtil.transpose_A_times_X(g, a);
      try (TInt32 x = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
        x.set(5, 0, 0);
        x.set(7, 1, 0);
        try (TestUtil.AutoCloseableList<Tensor<?>> outputs =
              new TestUtil.AutoCloseableList<>(s.runner().feed("X", x).fetch("Y").run())) {
          assertEquals(1, outputs.size());
          assertEquals(31, outputs.get(0).get(0, 0));
        }
      }
    }
  }

  @Test
  public void runUsingOperationHandles() {
    try (Graph g = new Graph();
        Session s = new Session(g);
        TInt32 a = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
      a.set(2, 0, 0);
      a.set(3, 0, 0);
      TestUtil.transpose_A_times_X(g, a);
      Output<TInt32> feed = g.operation("X").output(0);
      Output<TInt32> fetch = g.operation("Y").output(0);
      try (TInt32 x = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
        x.set(5, 0, 0);
        x.set(7, 1, 0);
        try (TestUtil.AutoCloseableList<Tensor<?>> outputs =
            new TestUtil.AutoCloseableList<Tensor<?>>(
                s.runner().feed(feed, x).fetch(fetch).run())) {
          assertEquals(1, outputs.size());
          assertEquals(31, outputs.get(0).get(0, 0));
        }
      }
    }
  }

  @Test
  public void runUsingColonSeparatedNames() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      Operation split =
          g.opBuilder("Split", "Split")
              .addInput(TestUtil.constant(g, "split_dim", Tensors.scalar(0)))
              .addInput(TestUtil.constant(g, "value", Tensors.vector(new int[] {1, 2, 3, 4})))
              .setAttr("num_split", 2)
              .build();
      g.opBuilder("Add", "Add")
          .addInput(split.output(0))
          .addInput(split.output(1))
          .build()
          .output(0);
      // Fetch using colon separated names.
      try (TInt32 fetched =
          s.runner().fetch("Split:1").run().get(0).expect(TInt32.DTYPE)) {
        assertEquals(3, fetched.get(0).intValue());
        assertEquals(4, fetched.get(1).intValue());
      }
      // Feed using colon separated names.
      try (TInt32 fed = Tensors.vector(new int[] {4, 3, 2, 1});
          TInt32 fetched =
              s.runner()
                  .feed("Split:0", fed)
                  .feed("Split:1", fed)
                  .fetch("Add")
                  .run()
                  .get(0)
                  .expect(TInt32.DTYPE)) {
        assertEquals(8, fetched.get(0).intValue());
        assertEquals(6, fetched.get(1).intValue());
        assertEquals(4, fetched.get(2).intValue());
        assertEquals(2, fetched.get(3).intValue());
      }
    }
  }

  @Test
  public void runWithMetadata() {
    try (Graph g = new Graph();
        Session s = new Session(g);
        TInt32 a = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
      a.set(2, 0, 0);
      a.set(3, 0, 0);
      TestUtil.transpose_A_times_X(g, a);
      try (TInt32 x = Tensors.of(TInt32.DTYPE, Shape.make(2, 1))) {
        x.set(5, 0, 0);
        x.set(7, 0, 0);
        Session.Run result =
            s.runner()
                .feed("X", x)
                .fetch("Y")
                .setOptions(fullTraceRunOptions())
                .runAndFetchMetadata();
        // Sanity check on outputs.
        TestUtil.AutoCloseableList<Tensor<?>> outputs = new TestUtil.AutoCloseableList<>(result.outputs);
        assertEquals(1, outputs.size());
        assertEquals(31, outputs.get(0).get(0, 0));
        // Sanity check on metadata
        // See comments in fullTraceRunOptions() for an explanation about
        // why this check is really silly. Ideally, this would be:
      /*
          RunMetadata md = RunMetadata.parseFrom(result.metadata);
          assertTrue(md.toString(), md.hasStepStats());
      */
        assertTrue(result.metadata.length > 0);
        outputs.close();
      }
    }
  }

  @Test
  public void runMultipleOutputs() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      TestUtil.constant(g, "c1", Tensors.scalar(2718));
      TestUtil.constant(g, "c2", Tensors.scalar(31415));
      TestUtil.AutoCloseableList<Tensor<?>> outputs =
          new TestUtil.AutoCloseableList<>(s.runner().fetch("c2").fetch("c1").run());
      assertEquals(2, outputs.size());
      assertEquals(Integer.valueOf(31415), outputs.get(0).expect(TInt32.DTYPE).get());
      assertEquals(Integer.valueOf(2718), outputs.get(1).expect(TInt32.DTYPE).get());
      outputs.close();
    }
  }

  @Test
  public void failOnUseAfterClose() {
    try (Graph g = new Graph()) {
      Session s = new Session(g);
      s.close();
      try {
        s.runner().run();
        fail("methods on a session should fail after close() is called");
      } catch (IllegalStateException e) {
        // expected exception
      }
    }
  }

  @Test
  public void createWithConfigProto() {
    try (Graph g = new Graph();
        Session s = new Session(g, singleThreadConfigProto())) {}
  }

  private static byte[] fullTraceRunOptions() {
    // Ideally this would use the generated Java sources for protocol buffers
    // and end up with something like the snippet below. However, generating
    // the Java files for the .proto files in tensorflow/core:protos_all is
    // a bit cumbersome in bazel until the proto_library rule is setup.
    //
    // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
    //
    // For this test, for now, the use of specific bytes suffices.
    return new byte[] {0x08, 0x03};
    /*
    return org.tensorflow.framework.RunOptions.newBuilder()
        .setTraceLevel(RunOptions.TraceLevel.FULL_TRACE)
        .build()
        .toByteArray();
    */
  }

  public static byte[] singleThreadConfigProto() {
    // Ideally this would use the generated Java sources for protocol buffers
    // and end up with something like the snippet below. However, generating
    // the Java files for the .proto files in tensorflow/core:protos_all is
    // a bit cumbersome in bazel until the proto_library rule is setup.
    //
    // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
    //
    // For this test, for now, the use of specific bytes suffices.
    return new byte[] {0x10, 0x01, 0x28, 0x01};
    /*
    return org.tensorflow.framework.ConfigProto.newBuilder()
        .setInterOpParallelismThreads(1)
        .setIntraOpParallelismThreads(1)
        .build()
        .toByteArray();
     */
  }
}
