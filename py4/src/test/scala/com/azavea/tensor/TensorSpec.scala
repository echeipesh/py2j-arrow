package com.azavea.tensor

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import org.apache.arrow.flatbuf.Tensor
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ReadChannel}
import org.apache.arrow.vector.ipc.message.MessageSerializer
import org.nd4j.arrow.ArrowSerde
import org.scalatest._

class TensorSpec extends FunSuite with Matchers {
//  val bytes = Files.readAllBytes(path)

  test("decode tensor IPC message") {
    import org.apache.arrow.memory.RootAllocator
    import java.io.FileInputStream
    val path = Paths.get("/Users/eugene/tmp/pyarrow-tensor-ipc-roundtrip")
    val inFile = path.toFile
    val allocator = new RootAllocator(Integer.MAX_VALUE)
    val fileInputStream = new FileInputStream(inFile)
    val channel = fileInputStream.getChannel
    val readChannel = new ReadChannel(channel)
    val msg = MessageSerializer.readMessage(readChannel)
    val body = MessageSerializer.readMessageBody(readChannel, msg.getMessageLength, allocator)
    val tensor = new Tensor()
    info(msg.toString)
    info(s"msg length: ${msg.getMessage.header(tensor)}")
    info("header type: " + msg.getMessage.headerType().toString)
    info("tensor: " + tensor)
    info("stridesLength:" + tensor.stridesLength())
    info("strides(0)" + tensor.strides(0))
    info("strides(1)" + tensor.strides(1))


    info(body.toString)


    val first = body.getDouble(0)
    println(s"first = $first")

//    val tens = Tensor.getRootAsTensor(ByteBuffer.wrap(bytes))
//    val n = ArrowSerde.fromTensor(tens)
//    info(n.toString)
  }
}