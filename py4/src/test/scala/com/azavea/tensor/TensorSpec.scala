package com.azavea.tensor

import java.io.{ByteArrayOutputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.azavea.table.IntArrowTile
import com.google.flatbuffers.FlatBufferBuilder
import org.apache.arrow.flatbuf.{Tensor, TensorDim, Type}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ReadChannel, WriteChannel}
import org.apache.arrow.vector.ipc.message.MessageSerializer
import org.nd4j.arrow.ArrowSerde
import org.scalatest._

class TensorSpec extends FunSuite with Matchers {
  test("decode tensor IPC message") {
    import org.apache.arrow.memory.RootAllocator
    import java.io.FileInputStream
    /*
    [[ 1.99082191 -2.21780841 -0.33312663 -0.44599177]
     [-1.17051012  1.02869472 -0.47140698 -2.00142025]
     [ 0.53426529 -1.42411179  1.16711479 -0.08009347]
     [-0.24425364 -0.30497303  1.30974274 -1.5765599 ]
     [-0.58465982  0.17825995  0.96641421 -0.16277784]
     [ 1.28418918 -0.47113402  0.2801445  -2.07823133]
     [-1.04648874 -0.18688119  2.68314044 -0.30744865]
     [ 0.49801276 -0.51182441  0.00787912 -0.49916398]
     [ 0.66267546 -0.31091856 -0.68961703  0.82970454]
     [ 0.05621398  1.84742877 -1.08983915 -1.60066008]]
     */
    val path = Paths.get("/Users/eugene/tmp/pyarrow-tensor-ipc-roundtrip")
    val inFile = path.toFile
    val allocator = new RootAllocator(Integer.MAX_VALUE)
    val fileInputStream = new FileInputStream(inFile)
    val channel = fileInputStream.getChannel
    val readChannel = new ReadChannel(channel)

    val msg = MessageSerializer.readMessage(readChannel)
    info(s"msg length: ${msg.getMessageLength}")
    info(s"msg body length: ${msg.getMessageBodyLength}")
    info(s"msg type: ${msg.headerType()}")
    info(s"msg has body: ${msg.messageHasBody()}")

    val tensor = new Tensor()
    msg.getMessage.header(tensor)

    val typeStruct = new org.apache.arrow.flatbuf.Decimal()
    tensor.`type`(typeStruct)
    info(s"tensor type: precision: ${typeStruct.precision()}  scale: ${typeStruct.scale()}")

    info("tensor: " + tensor)
    info("stridesLength:" + tensor.stridesLength())
    info("strides(0)" + tensor.strides(0))
    info("strides(1)" + tensor.strides(1))
    info("shapeLength()" + tensor.shapeLength())
    info("shape(0): " + tensor.shape(0).size) // 10
    info("shape(1): " + tensor.shape(1).size) // 4
    info(s"data offset: " + tensor.data().offset())
    info(s"data size:" + tensor.data().length())

    val body = MessageSerializer.readMessageBody(readChannel, msg.getMessageLength, allocator)
    val first = body.getDouble(0)
    info(s"first = $first")
    info(s"second = " + body.getDouble(1))

  }

  test("decode own tensor IPC message") {
    import org.apache.arrow.memory.RootAllocator
    import java.io.FileInputStream
    val path = Paths.get("/Users/eugene/tmp/tensor-out.np")
    val inFile = path.toFile
    val allocator = new RootAllocator(Integer.MAX_VALUE)
    val fileInputStream = new FileInputStream(inFile)
    val channel = fileInputStream.getChannel
    val readChannel = new ReadChannel(channel)
    val msg = MessageSerializer.readMessage(readChannel)

    info(s"msg length: ${msg.getMessageLength}")
    info(s"msg body length: ${msg.getMessageBodyLength}")
    info(s"msg type: ${msg.headerType()}")
    info(s"msg has body: ${msg.messageHasBody()}")

    val tensor = new Tensor()
    msg.getMessage.header(tensor)

    val typeStruct = new org.apache.arrow.flatbuf.Int()
    tensor.`type`(typeStruct)

    info("tensor: " + tensor)
    info("tensor type: " + typeStruct.bitWidth() + " " + typeStruct.isSigned)
    info("stridesLength:" + tensor.stridesLength())
    info("strides(0)" + tensor.strides(0))
    info("strides(1)" + tensor.strides(1))
    info("shapeLength()" + tensor.shapeLength())
    info("shape(0): " + tensor.shape(0).size) // 10
    info("shape(1): " + tensor.shape(1).size) // 4
    info("type: " + tensor.typeType())
    info(s"data offset: " + tensor.data().offset())
    info(s"data size:" + tensor.data().length())
    val body = MessageSerializer.readMessageBody(readChannel, msg.getMessageBodyLength.toInt, allocator)
    info(s"first  = " + body.getInt(0))
    info(s"second = " + body.getInt(1))
  }


  test("Init ArrowCube from Tensor IPC message") {
    // ArrowCube has a buffer
    // Arrow Cube has dimension information
  }

  test("IntArrayTile to Tensor") {
    // lets just go medieval on this, a function was good enough for my grandpa, its good enough for me.
    val data = Array[Int](0, 1, 2, 3)
    // TODO: Note that we get 16777216 as second number
    val tile = IntArrowTile(data, 2, 2)
    val bb = tile.toIpcMessage()
    import java.nio.channels.Channels
    import java.nio.channels.WritableByteChannel
    val fos = new FileOutputStream("/Users/eugene/tmp/tensor-out.np")
    val wbc = new WriteChannel(Channels.newChannel(fos))
    info(s"message buffer remaining: ${bb.remaining()}")
    MessageSerializer.writeMessageBuffer(wbc, bb.remaining(), bb)
    wbc.align()
    info("wrote first: " + tile.array.getDataBuffer.getInt(0))
    info("wrote secnd: " + tile.array.getDataBuffer.getInt(1))
    info("data buffer capacity: " + tile.array.getDataBuffer.capacity())
    wbc.align()
    wbc.write(tile.array.getDataBuffer)
    wbc.close()
    fos.flush()
    fos.close()
    /*
    <pyarrow.Tensor>
    type: int32
    shape: (2, 2)
    strides: (8, 4)
    [[0 1]
     [2 3]]
   */

  }
}