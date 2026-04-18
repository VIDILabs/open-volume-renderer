if(NOT DEFINED GIT_EXECUTABLE OR GIT_EXECUTABLE STREQUAL "")
  message(FATAL_ERROR "GIT_EXECUTABLE is required")
endif()

if(NOT DEFINED IMGUI_PATCH OR IMGUI_PATCH STREQUAL "")
  message(FATAL_ERROR "IMGUI_PATCH is required")
endif()

if(NOT DEFINED IMGUI_SOURCE_DIR OR IMGUI_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "IMGUI_SOURCE_DIR is required")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check "${IMGUI_PATCH}"
  WORKING_DIRECTORY "${IMGUI_SOURCE_DIR}"
  RESULT_VARIABLE _patch_check_result
  OUTPUT_QUIET
  ERROR_QUIET
)

if(_patch_check_result EQUAL 0)
  message(STATUS "Applying ImGui patch: ${IMGUI_PATCH}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply "${IMGUI_PATCH}"
    WORKING_DIRECTORY "${IMGUI_SOURCE_DIR}"
    RESULT_VARIABLE _patch_apply_result
    OUTPUT_VARIABLE _patch_apply_out
    ERROR_VARIABLE _patch_apply_err
  )
  if(NOT _patch_apply_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to apply ImGui patch.\n"
      "stdout:\n${_patch_apply_out}\n"
      "stderr:\n${_patch_apply_err}"
    )
  endif()
  return()
endif()

# If the forward check fails, accept the "already applied" case.
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${IMGUI_PATCH}"
  WORKING_DIRECTORY "${IMGUI_SOURCE_DIR}"
  RESULT_VARIABLE _patch_reverse_check_result
  OUTPUT_QUIET
  ERROR_QUIET
)

if(_patch_reverse_check_result EQUAL 0)
  message(STATUS "ImGui patch already applied, skipping")
  return()
endif()

message(FATAL_ERROR
  "ImGui patch is neither applicable nor already applied.\n"
  "Patch: ${IMGUI_PATCH}\n"
  "Source: ${IMGUI_SOURCE_DIR}"
)
