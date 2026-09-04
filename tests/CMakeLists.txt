add_executable(rt_tests
  test_bf16.cpp
  test_mat.cpp
  test_quant.cpp
  test_gguf.cpp
  test_ndarray.cpp
  test_tensor_node.cpp
  test_nn2.cpp
)
target_link_libraries(rt_tests PRIVATE rt_core Catch2::Catch2WithMain)
catch_discover_tests(rt_tests)
