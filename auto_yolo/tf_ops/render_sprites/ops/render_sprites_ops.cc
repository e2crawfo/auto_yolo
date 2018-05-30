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
    .Input("sprites: T")
    .Input("n_sprites: int32")
    .Input("scales: T")
    .Input("offsets: T")
    .Input("backgrounds: T")

    .Output("output: T")

    .Attr("T: {float}")
    // .Attr("T: {half, float, double}")

    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle sprites;
      ShapeHandle n_sprites;
      ShapeHandle scales;
      ShapeHandle offsets;
      ShapeHandle backgrounds;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &sprites));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &n_sprites));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &scales));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &offsets));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &backgrounds));

      c->set_output(0, backgrounds);
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(RenderSprites op.)doc");

REGISTER_OP("RenderSpritesGrad")
    .Input("sprites: T")
    .Input("n_sprites: int32")
    .Input("scales: T")
    .Input("offsets: T")
    .Input("backgrounds: T")
    .Input("grad_output: T")

    .Output("grad_sprites: T")
    .Output("grad_n_sprites: T")
    .Output("grad_scales: T")
    .Output("grad_offsets: T")
    .Output("grad_backgrounds: T")

    .Attr("T: {float}")
    // .Attr("T: {half, float, double}")

    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle sprites;
      ShapeHandle n_sprites;
      ShapeHandle scales;
      ShapeHandle offsets;
      ShapeHandle backgrounds;
      ShapeHandle grad_output;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &sprites));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &n_sprites));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &scales));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &offsets));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &backgrounds));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 4, &grad_output));

      c->set_output(0, sprites);
      c->set_output(1, n_sprites);
      c->set_output(2, scales);
      c->set_output(3, offsets);
      c->set_output(4, backgrounds);

      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(RenderSprites Grad op.)doc");

}  // namespace tensorflow
