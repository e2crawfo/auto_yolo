// Copyright 2017 The Sonnet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RenderSprites")
    .Input("sprites: N * T")
    .Input("scales: N * T")
    .Input("offsets: N * T")
    .Input("backgrounds: T")

    .Output("output: T")

    .Attr("N: int")
    .Attr("T: {float}")

    .SetShapeFn([](InferenceContext* c) {
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for(int i=0; i < N; i++){
        ShapeHandle sprites_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &sprites_shape));
        ShapeHandle scales_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(N+i), 3, &scales_shape));
        ShapeHandle offsets_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2*N+i), 3, &offsets_shape));
      }

      ShapeHandle backgrounds_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3*N), 4, &backgrounds_shape));

      c->set_output(0, backgrounds_shape);

      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(RenderSprites op.)doc");

REGISTER_OP("RenderSpritesGrad")
    .Input("sprites: N * T")
    .Input("scales: N * T")
    .Input("offsets: N * T")
    .Input("backgrounds: T")
    .Input("grad_output: T")

    .Output("grads: M * T")

    .Attr("N: int")
    .Attr("M: int")
    .Attr("T: {float}")

    .SetShapeFn([](InferenceContext* c) {
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for(int i=0; i < N; i++){
        ShapeHandle sprites_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 5, &sprites_shape));
        c->set_output(i, sprites_shape);

        ShapeHandle scales_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(N+i), 3, &scales_shape));
        c->set_output(N+i, scales_shape);

        ShapeHandle offsets_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2*N+i), 3, &offsets_shape));
        c->set_output(2*N+i, offsets_shape);
      }

      ShapeHandle backgrounds_shape, grad_output_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3*N), 4, &backgrounds_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3*N+1), 4, &grad_output_shape));

      c->set_output(3*N, backgrounds_shape);

      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(RenderSprites Grad op.)doc");

}  // namespace tensorflow
