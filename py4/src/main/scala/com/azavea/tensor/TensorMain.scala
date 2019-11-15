package com.azavea.tensor

import java.io.File
import java.nio.file.{Files, Path, Paths}

object TensorMain extends App {

  val path = Paths.get("/Users/eugene/tmp/pyarrow-tensor-ipc-roundtrip")
  val bytes = Files.readAllBytes(path)

}
