# Turn an .msl file into a C++ header holding it as a raw string literal, so the
# shaders stay editable as MSL (syntax highlighting, external validation) while
# still compiling into the binary with no runtime file lookup.
function(rt_embed_shader MSL_PATH OUT_HEADER SYMBOL)
  file(READ "${MSL_PATH}" _msl_source)
  # RT_MSL is an unlikely delimiter to appear in shader source, so the raw
  # string cannot be terminated early.
  set(_content "// Generated from ${MSL_PATH}. Do not edit; edit the .msl instead.\n")
  string(APPEND _content "#pragma once\n\nnamespace rt {\n\n")
  string(APPEND _content "inline constexpr const char* ${SYMBOL} = R\"RT_MSL(\n")
  string(APPEND _content "${_msl_source}")
  string(APPEND _content ")RT_MSL\";\n\n}  // namespace rt\n")
  file(GENERATE OUTPUT "${OUT_HEADER}" CONTENT "${_content}")
endfunction()
