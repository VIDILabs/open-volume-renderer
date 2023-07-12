// ======================================================================== //
// Copyright Qi Wu, since 2019                                              //
// Copyright SCI Institute, University of Utah, 2018                        //
// ======================================================================== //
#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "core.h"
#include "default.h"

#include <imconfig.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tfn {

class TFN_MODULE_INTERFACE TransferFunctionWidget
{
 private:
  using setter = std::function<void(const list3f &, const list2f &, const vec2f &)>;

 private:
  /* Variables Controlled by Users */
  setter _setter_cb;
  vec2f valueRange; //< the current value range controlled by the user
  vec2f defaultRange; //< the default value range being displayed on the GUI

  /* The 2d palette texture on the GPU for displaying the color map preview in the UI. */
  GLuint tfn_palette;

  // all available transfer functions
  std::vector<std::string> tfns_names;
  std::vector<tfn::TransferFunctionCore> tfns;

  using ColorPoint = tfn::TransferFunctionCore::ColorControl;
  using AlphaPoint = tfn::TransferFunctionCore::AlphaControl;
  using GaussianPoint = tfn::TransferFunctionCore::GaussianObject;

  // properties of currently selected transfer function
  int tfn_selection{-1};
  std::vector<ColorPoint>* current_colorpoints{};
  std::vector<AlphaPoint>* current_alphapoints{};
  vec2i current_tfn_editable{1, 1};

  std::vector<AlphaPoint> uneditable_alphapoints;

  // flag indicating transfer function has changed in UI
  bool tfn_changed{true};
  bool tfn_applied{true};

  // scaling factor for generated alphas
  float global_alpha_scale{1.f};

  // domain (value range) of transfer function
  vec2f value_range{-1.f, 1.f};
  vec2f value_range_default{-1.f, 1.f};
  vec2f value_range_percentage{0.f, 100.f};

  // The filename input text buffer
  std::vector<char> tfn_text_buffer; 

 public:
  ~TransferFunctionWidget();
  TransferFunctionWidget(const setter &);
  
  /* Setup the default data value range for the transfer function */
  void set_default_value_range(const float &a, const float &b);
  
  /* Draw the transfer function editor widget, returns true if the transfer function changed */
  bool build(bool *p_open = NULL, bool do_not_render_textures = false);
  
  /* Construct the ImGui GUI */
  void build_gui();
  
  /* Render the transfer function to a 1D texture that can be applied to volume data */
  void render(int tfn_w = 256, int tfn_h = 1);

  /* Load the transfer function in the file passed and set it active */
  void load(const std::string &fileName);

  /* Save the current transfer function out to the file */
  void save(const std::string &fileName) const;

  /* Create a new TFN profile */
  // void add_tfn(const tfn::TransferFunctionCore& core, const std::string &name);
  void add_tfn(const list4f &, const list2f &, const std::string &name);
  
 private:
  /* Change selection */
  void select_tfn(int selection);
  /** Load all the pre-defined transfer function maps */
  void set_default_tfns();
  /** Draw the Tfn Editor in a window */
  void draw_tfn_editor(const float margin, const float height);
  tfn::vec4f draw_tfn_editor__preview_texture(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &);
  tfn::vec4f draw_tfn_editor__color_control_points(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &);
  tfn::vec4f draw_tfn_editor__alpha_control_points(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &);
  tfn::vec4f draw_tfn_editor__interaction_blocks(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &, const float &);
};

inline void TransferFunctionWidget::select_tfn(int selection)
{
  if (tfn_selection != selection) 
  {
    tfn_selection = selection;

    auto& tfn = tfns[tfn_selection];

    current_colorpoints = tfn.colorControlVector();
    current_tfn_editable.x = (tfn.colorControlCount() > 128) ? 0 : 1;

    // in this case we have to use the raw RGBA table
    if (tfn.alphaControlCount() == 0 || tfn.gaussianObjectCount() > 0) 
    {
      uneditable_alphapoints.resize(tfn.resolution());
      const auto *table = (vec4f *)tfn.data();
      for (int i = 0; i < uneditable_alphapoints.size(); ++i) {
        uneditable_alphapoints[i] = vec2f((float)i / (uneditable_alphapoints.size() - 1), table[i].w);
      }
      current_alphapoints = &uneditable_alphapoints;
      current_tfn_editable.y = 0;
    }
    else {
      current_alphapoints = tfn.alphaControlVector();
      current_tfn_editable.y = (tfn.alphaControlCount() > 128) ? 0 : 1;
    }

    tfn_changed  = true;
  }
}

inline TransferFunctionWidget::~TransferFunctionWidget()
{
  if (tfn_palette) glDeleteTextures(1, &tfn_palette);
}

inline TransferFunctionWidget::TransferFunctionWidget(const setter &fcn)
    : tfn_changed(true), tfn_palette(0), tfn_text_buffer(512, '\0'), _setter_cb(fcn), valueRange{0.f, 0.f}, defaultRange{0.f, 0.f}
{
  set_default_tfns();
  select_tfn(0);
}

// inline void TransferFunctionWidget::add_tfn(const tfn::TransferFunctionCore& core, const std::string &name)
// {
//   auto it = std::find(tfns_names.begin(), tfns_names.end(), name);
//   if (it == tfns_names.end()) {
//     tfns.push_back(core);
//     tfns.back().updateColorMap();
//     tfns_names.push_back(name);
//     select_tfn((int)(tfns.size() - 1)); // Remember to update other constructors also
//   } else {
//     select_tfn((int)std::distance(tfns_names.begin(), it));
//   }
// }

inline void TransferFunctionWidget::add_tfn(const list4f &ct, const list2f &ot, const std::string &name)
{
  auto it = std::find(tfns_names.begin(), tfns_names.end(), name);

  if (it == tfns_names.end()) {
    tfns.emplace_back();
    auto& tfn = tfns.back();

    for (size_t i = 0; i < ct.size(); ++i) {
      tfn.addColorControl(ct[i].x, ct[i].y, ct[i].z, ct[i].w);
    }

    for (size_t i = 0; i < ot.size(); ++i) {
      tfn.addAlphaControl(vec2f{ot[i].x, ot[i].y});
    }
    
    tfn.updateColorMap();

    tfns_names.push_back(name);

    select_tfn((int)(tfns.size() - 1)); // Remember to update other constructors also
  } else {
    select_tfn((int)std::distance(tfns_names.begin(), it));
  }
}

inline void TransferFunctionWidget::set_default_value_range(const float &a, const float &b)
{
  if (b >= a) {
    valueRange.x = defaultRange.x = a;
    valueRange.y = defaultRange.y = b;
    tfn_changed = true;
  }
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__preview_texture(void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor)
{
  auto draw_list = (ImDrawList *)_draw_list;
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y));
  ImGui::Image(reinterpret_cast<void *>(tfn_palette), (const ImVec2 &)size);
  ImGui::SetCursorScreenPos((const ImVec2 &)cursor);
  // TODO: more generic way of drawing arbitary splats
  for (int i = 0; i < current_alphapoints->size() - 1; ++i) {
    std::vector<ImVec2> polyline;
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i].pos.x * size.x, cursor.y + size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i].pos.x * size.x, cursor.y + (1.f - (*current_alphapoints)[i].pos.y) * size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i + 1].pos.x * size.x + 1, cursor.y + (1.f - (*current_alphapoints)[i + 1].pos.y) * size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i + 1].pos.x * size.x + 1, cursor.y + size.y);
#ifdef IMGUI_VERSION_NUM
    draw_list->AddConvexPolyFilled(polyline.data(), (int)polyline.size(), 0xFFD8D8D8 /*, true*/);
#else
    draw_list->AddConvexPolyFilled(polyline.data(), (int)polyline.size(), 0xFFD8D8D8, true);
#endif
  }
  tfn::vec4f new_cursor = {
      cursor.x,
      cursor.y + size.y + margin.z,
      cursor.z,
      cursor.w - size.y,
  };
  ImGui::SetCursorScreenPos((const ImVec2 &)new_cursor);
  return new_cursor;
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__color_control_points(void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &color_len)
{
  auto draw_list = (ImDrawList *)_draw_list;
  // draw circle background
  draw_list->AddRectFilled(
      ImVec2(cursor.x + margin.x, cursor.y - margin.z), 
      ImVec2(cursor.x + margin.x + size.x, cursor.y - margin.x + 2.5f * color_len),
      0xFF474646
  );
  // draw circles
  for (int i = (int)current_colorpoints->size() - 1; i >= 0; --i) {
    const ImVec2 pos(cursor.x + size.x * (*current_colorpoints)[i].position + margin.x, cursor.y);
    ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y));
    // white background
    draw_list->AddTriangleFilled(
        ImVec2(pos.x - 0.5f * color_len, pos.y), ImVec2(pos.x + 0.5f * color_len, pos.y), ImVec2(pos.x, pos.y - color_len), 0xFFD8D8D8);
    draw_list->AddCircleFilled(ImVec2(pos.x, pos.y + 0.5f * color_len), color_len, 0xFFD8D8D8);
    // draw picker
    ImVec4 picked_color = ImColor((*current_colorpoints)[i].color.x, (*current_colorpoints)[i].color.y, (*current_colorpoints)[i].color.z, 1.f);
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y + 1.5f * color_len));
    if (ImGui::ColorEdit4(("##ColorPicker" + std::to_string(i)).c_str(),
            (float *)&picked_color,
            ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview
                | ImGuiColorEditFlags_NoOptions | ImGuiColorEditFlags_NoTooltip)) {
      (*current_colorpoints)[i].color.x = picked_color.x;
      (*current_colorpoints)[i].color.y = picked_color.y;
      (*current_colorpoints)[i].color.z = picked_color.z;
      tfn_changed = true;
    }
    if (ImGui::IsItemHovered()) {
      // convert float color to char
      int cr = static_cast<int>(picked_color.x * 255);
      int cg = static_cast<int>(picked_color.y * 255);
      int cb = static_cast<int>(picked_color.z * 255);
      // setup tooltip
      ImGui::BeginTooltip();
      ImVec2 sz(ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2, ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2);
      ImGui::ColorButton("##PreviewColor", picked_color, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_AlphaPreview, sz);
      ImGui::SameLine();
      ImGui::Text(
          "Left click to edit\n"
          "HEX: #%02X%02X%02X\n"
          "RGB: [%3d,%3d,%3d]\n(%.2f, %.2f, %.2f)",
          cr,
          cg,
          cb,
          cr,
          cg,
          cb,
          picked_color.x,
          picked_color.y,
          picked_color.z);
      ImGui::EndTooltip();
    }
  }
  for (int i = 0; i < current_colorpoints->size(); ++i) {
    const ImVec2 pos(cursor.x + size.x * (*current_colorpoints)[i].position + margin.x, cursor.y);
    // draw button
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y - 0.5f * color_len));
    ImGui::InvisibleButton(("##ColorControl-" + std::to_string(i)).c_str(), ImVec2(2.f * color_len, 2.f * color_len));
    // dark highlight
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y));
    draw_list->AddCircleFilled(ImVec2(pos.x, pos.y + 0.5f * color_len), 0.5f * color_len, ImGui::IsItemHovered() ? 0xFF051C33 : 0xFFBCBCBC);
    // delete color point
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
      if (i > 0 && i < current_colorpoints->size() - 1) {
        current_colorpoints->erase(current_colorpoints->begin() + i);
        tfn_changed = true;
      }
    }
    // drag color control point
    else if (ImGui::IsItemActive()) {
      ImVec2 delta = ImGui::GetIO().MouseDelta;
      if (i > 0 && i < current_colorpoints->size() - 1) {
        (*current_colorpoints)[i].position += delta.x / size.x;
        (*current_colorpoints)[i].position = clamp((*current_colorpoints)[i].position, (*current_colorpoints)[i - 1].position, (*current_colorpoints)[i + 1].position);
      }
      tfn_changed = true;
    }
  }
  return vec4f();
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__alpha_control_points(/**/
    void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &alpha_len)
{
  auto draw_list = (ImDrawList *)_draw_list;
  // draw circles
  for (int i = 0; i < current_alphapoints->size(); ++i) {
    const ImVec2 pos(cursor.x + size.x * (*current_alphapoints)[i].pos.x + margin.x, cursor.y - size.y * (*current_alphapoints)[i].pos.y - margin.z);
    ImGui::SetCursorScreenPos(ImVec2(pos.x - alpha_len, pos.y - alpha_len));
    ImGui::InvisibleButton(("##AlphaControl-" + std::to_string(i)).c_str(), ImVec2(2.f * alpha_len, 2.f * alpha_len));
    ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y));
    // dark bounding box
    draw_list->AddCircleFilled(pos, alpha_len, 0xFF565656);
    // white background
    draw_list->AddCircleFilled(pos, 0.8f * alpha_len, 0xFFD8D8D8);
    // highlight
    draw_list->AddCircleFilled(pos, 0.6f * alpha_len, ImGui::IsItemHovered() ? 0xFF051c33 : 0xFFD8D8D8);
    // delete alpha point
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
      if (i > 0 && i < current_alphapoints->size() - 1) {
        current_alphapoints->erase(current_alphapoints->begin() + i);
        tfn_changed = true;
      }
    } 
    // drag alpha control point
    else if (ImGui::IsItemActive()) {
      ImVec2 delta = ImGui::GetIO().MouseDelta;
      (*current_alphapoints)[i].pos.y -= delta.y / size.y;
      (*current_alphapoints)[i].pos.y = clamp((*current_alphapoints)[i].pos.y, 0.0f, 1.0f);
      if (i > 0 && i < current_alphapoints->size() - 1) {
        (*current_alphapoints)[i].pos.x += delta.x / size.x;
        (*current_alphapoints)[i].pos.x = clamp((*current_alphapoints)[i].pos.x, (*current_alphapoints)[i - 1].pos.x, (*current_alphapoints)[i + 1].pos.x);
      }
      tfn_changed = true;
    }
  }
  return vec4f();
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__interaction_blocks(/**/
    void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing */
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &color_len,
    const float &alpha_len)
{
  const float mouse_x = ImGui::GetMousePos().x;
  const float mouse_y = ImGui::GetMousePos().y;
  const float scroll_x = ImGui::GetScrollX();
  const float scroll_y = ImGui::GetScrollY();
  auto draw_list = (ImDrawList *)_draw_list;
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y - margin.z));
  ImGui::InvisibleButton("##tfn_palette_color", ImVec2(size.x, 2.5f * color_len));
  // add color point
  if (current_tfn_editable.x && ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
    const float p = clamp((mouse_x - cursor.x - margin.x - scroll_x) / (float)size.x, 0.f, 1.f);
    int il, ir;
    std::tie(il, ir) = find_interval(current_colorpoints, p);
    const float pr = (*current_colorpoints)[ir].position;
    const float pl = (*current_colorpoints)[il].position;
    const float r = lerp((*current_colorpoints)[il].color.x, (*current_colorpoints)[ir].color.x, pl, pr, p);
    const float g = lerp((*current_colorpoints)[il].color.y, (*current_colorpoints)[ir].color.y, pl, pr, p);
    const float b = lerp((*current_colorpoints)[il].color.z, (*current_colorpoints)[ir].color.z, pl, pr, p);
    ColorPoint pt;
    pt.position = p, pt.color.x = r, pt.color.y = g, pt.color.z = b;
    current_colorpoints->insert(current_colorpoints->begin() + ir, pt);
    tfn_changed = true;
  }
  // draw background interaction
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y - size.y - margin.z));
  if (size.x > 0 && size.y > 0) ImGui::InvisibleButton("##tfn_palette_alpha", ImVec2(size.x, size.y));
  // add alpha point
  if (current_tfn_editable.y && ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
    const float x = clamp((mouse_x - cursor.x - margin.x - scroll_x) / (float)size.x, 0.f, 1.f);
    const float y = clamp(-(mouse_y - cursor.y + margin.x - scroll_y) / (float)size.y, 0.f, 1.f);
    int il, ir;
    std::tie(il, ir) = find_interval(current_alphapoints, x);
    AlphaPoint pt;
    pt.pos.x = x, pt.pos.y = y;
    current_alphapoints->insert(current_alphapoints->begin() + ir, pt);
    tfn_changed = true;
  }
  return vec4f();
}

inline void TransferFunctionWidget::draw_tfn_editor(const float margin, const float height)
{
  // style
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const float canvas_x = ImGui::GetCursorScreenPos().x;
  float canvas_y = ImGui::GetCursorScreenPos().y;
  const float width = ImGui::GetContentRegionAvail().x - 2.f * margin;
  const float color_len = 10.f;
  const float alpha_len = 10.f;
  // debug
  const tfn::vec3f m{margin, margin, margin};
  const tfn::vec2f s{width, height};
  tfn::vec4f c = {canvas_x, canvas_y, ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  // draw preview texture
  c = draw_tfn_editor__preview_texture(draw_list, m, s, c);
  canvas_y = c.y;
  // draw color control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  if (current_tfn_editable.x) {
    draw_tfn_editor__color_control_points(draw_list, m, s, c, color_len);
  }
  // draw alpha control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  if (current_tfn_editable.y) {
    draw_tfn_editor__alpha_control_points(draw_list, m, s, c, alpha_len);
  }
  // draw background interaction
  draw_tfn_editor__interaction_blocks(draw_list, m, s, c, color_len, alpha_len);
  // update cursors
  canvas_y += 4.f * color_len + margin;
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
}

inline bool TransferFunctionWidget::build(bool *p_open, bool do_not_render_textures)
{
  // ImGui::ShowTestWindow();

  ImGui::SetNextWindowSizeConstraints(ImVec2(400, 250), ImVec2(FLT_MAX, FLT_MAX));

  if (!ImGui::Begin("Transfer Function Widget", p_open)) {
    ImGui::End();
    return false;
  }

  build_gui();

  ImGui::End();

  if (!do_not_render_textures)
    render();

  return true;
}

void TransferFunctionWidget::build_gui()
{
  //------------ Styling ------------------------------
  const float margin = 10.f;

  //------------ Basic Controls -----------------------
  ImGui::Spacing();
  ImGui::SetCursorPosX(margin);
  ImGui::BeginGroup();
  {
    // /* title */
    // ImGui::Text("1D Transfer Function Editor");
    // ImGui::SameLine();
    // {
    //   ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.f);
    //   ImGui::Button("help");
    //   if (ImGui::IsItemHovered()) {
    //     ImGui::SetTooltip(
    //         "Double right click a control point to delete it\n"
    //         "Single left click and drag a control point to move it\n"
    //         "Double left click on an empty area to add a control point\n");
    //   }
    //   ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.f);
    // }
    // ImGui::Spacing();

    /* load a transfer function from file */
    ImGui::InputText("", tfn_text_buffer.data(), tfn_text_buffer.size() - 1);
    ImGui::SameLine();
    if (ImGui::Button("load tfn")) {
      try {
        std::string s = tfn_text_buffer.data();
        s.erase(s.find_last_not_of(" \n\r\t") + 1);
        s.erase(0, s.find_first_not_of(" \n\r\t"));
        load(s.c_str());
      } catch (const std::runtime_error &error) {
        std::cerr << "\033[1;33m" << "Error: " << error.what() << "\033[0m" << std::endl;
      }

      tfn_text_buffer = std::vector<char>(512, '\0');
    }

    // save function is not implemented
    ImGui::SameLine();
    if (ImGui::Button("save")) { 
      save(tfn_text_buffer.data()); 
      tfn_text_buffer = std::vector<char>(512, '\0');
    }

    /* Built-in color lists */
    {
      static int curr_tfn = tfn_selection;
      static std::string curr_names = "";
      curr_names = "";
      for (auto &n : tfns_names) {
        curr_names += n + '\0';
      }
      if (ImGui::Combo(" color tables", &curr_tfn, curr_names.c_str())) {
        select_tfn(curr_tfn);
      }
    }

    /* Display transfer function value range */
    static vec2f value_range_percentage(0.f, 100.f);
    if (defaultRange.y > defaultRange.x) {
      ImGui::Text(" default value range (%.6f, %.6f)", defaultRange.x, defaultRange.y);
      ImGui::Text(" current value range (%.6f, %.6f)", valueRange.x, valueRange.y);
      if (ImGui::DragFloat2(" value range %", (float *)&value_range_percentage, 1.f, 0.f, 100.f, "%.3f")) {
        tfn_changed = true;
        valueRange.x = (defaultRange.y - defaultRange.x) * value_range_percentage.x * 0.01f + defaultRange.x;
        valueRange.y = (defaultRange.y - defaultRange.x) * value_range_percentage.y * 0.01f + defaultRange.x;
      }
    }
  }

  ImGui::EndGroup();

  //------------ Transfer Function Editor -------------

  ImGui::Spacing();
  draw_tfn_editor(11.f, ImGui::GetContentRegionAvail().y - 60.f);

  //------------ End Transfer Function Editor ---------
}

inline void renderTFNTexture(GLuint &tex, int width, int height)
{
  GLint prev_binding = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_binding);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  if (prev_binding) {
    glBindTexture(GL_TEXTURE_2D, prev_binding);
  }
}

inline void TransferFunctionWidget::render(int tfn_w, int tfn_h)
{
  // Upload to GL if the transfer function has changed
  if (!tfn_palette) {
    renderTFNTexture(tfn_palette, tfn_w, tfn_h);
  } else {
    /* ... */
  }

  // Update texture color
  if (tfn_changed) {
    // Backup old states
    GLint prev_binding = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_binding);

    // Sample the palette then upload the data
    std::vector<uint8_t> palette(tfn_w * tfn_h * 4, 0);
    std::vector<vec3f> colors(tfn_w, 1.f);
    std::vector<vec2f> alpha(tfn_w, 1.f);
    const float step = 1.0f / (float)(tfn_w - 1);
    for (int i = 0; i < tfn_w; ++i) {
      const float p = clamp(i * step, 0.0f, 1.0f);
      int ir, il;
      /* color */
      {
        std::tie(il, ir) = find_interval(current_colorpoints, p);
        float pl = current_colorpoints->at(il).position;
        float pr = current_colorpoints->at(ir).position;
        const float r = lerp(current_colorpoints->at(il).color.x, current_colorpoints->at(ir).color.x, pl, pr, p);
        const float g = lerp(current_colorpoints->at(il).color.y, current_colorpoints->at(ir).color.y, pl, pr, p);
        const float b = lerp(current_colorpoints->at(il).color.z, current_colorpoints->at(ir).color.z, pl, pr, p);
        colors[i].x = r;
        colors[i].y = g;
        colors[i].z = b;
        /* palette */
        palette[i * 4 + 0] = static_cast<uint8_t>(r * 255.f);
        palette[i * 4 + 1] = static_cast<uint8_t>(g * 255.f);
        palette[i * 4 + 2] = static_cast<uint8_t>(b * 255.f);
        palette[i * 4 + 3] = 255;
      }
      /* alpha */
      {
        std::tie(il, ir) = find_interval(current_alphapoints, p);
        float pl = current_alphapoints->at(il).pos.x;
        float pr = current_alphapoints->at(ir).pos.x;
        const float a = lerp(current_alphapoints->at(il).pos.y, current_alphapoints->at(ir).pos.y, pl, pr, p);
        alpha[i].x = p;
        alpha[i].y = a;
      }
    }

    // Render palette again
    glBindTexture(GL_TEXTURE_2D, tfn_palette);
    glTexImage2D(GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        tfn_w,
        tfn_h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        static_cast<const void *>(palette.data())); // We need to resize texture of texture is resized
    if (prev_binding) { // Restore previous binded texture
      glBindTexture(GL_TEXTURE_2D, prev_binding);
    }

    this->_setter_cb(colors, alpha, valueRange);
    tfn_changed = false;
  }
}

inline void TransferFunctionWidget::load(const std::string &filename)
{
  TransferFunctionCore tfn;
  try {
    std::ifstream file(filename);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    json root = json::parse(text, nullptr, true, true);
    if (root.contains("view"))
      loadTransferFunction(root["view"]["volume"]["transferFunction"], tfn);
    else
      loadTransferFunction(root["transferFunction"], tfn);
  } 
  catch (...) {
    std::cout << "failed to load file: " << filename << std::endl;
    return;
  }

  tfns.push_back(std::move(tfn));
  tfns_names.push_back(filename);
  select_tfn((int)tfns.size() - 1);
}

inline void tfn::TransferFunctionWidget::save(const std::string &filename) const
{
  const auto& tfn = tfns[tfn_selection];

  json root = {{"transferFunction", {}}};
  saveTransferFunction(tfn, root["transferFunction"]);
  std::ofstream ofs(filename, std::ofstream::out);
  ofs << root.dump();
  ofs.close();
}

inline void TransferFunctionWidget::set_default_tfns()
{
  for (auto &ct : _predef_color_table_) {

    tfns.emplace_back();

    auto& tfn = tfns.back();

    for (size_t i = 0; i < ct.second.size() / 4; ++i) {
      tfn.addColorControl(ct.second[i * 4], ct.second[i * 4 + 1], ct.second[i * 4 + 2], ct.second[i * 4 + 3]);
    }

    tfn.addAlphaControl(vec2f(0.00f, 0.00f));
    tfn.addAlphaControl(vec2f(0.25f, 0.25f));
    tfn.addAlphaControl(vec2f(0.50f, 0.50f));
    tfn.addAlphaControl(vec2f(0.75f, 0.75f));
    tfn.addAlphaControl(vec2f(1.00f, 1.00f));
    tfn.updateColorMap();

    tfns_names.push_back(ct.first);
  }
};

} // namespace tfn
