package radiance.memory

import chisel3._
import chisel3.experimental.SourceInfo
import freechips.rocketchip.diplomacy.AddressSet
import freechips.rocketchip.tilelink.{TLAdapterNode, TLSlavePortParameters}
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.ValName
import org.chipsalliance.diplomacy.lazymodule._


class AddressRewriterNode(
  rewriteFn: UInt => UInt,
  inverseFn: UInt => UInt,
  managerRewriteFn: TLSlavePortParameters => TLSlavePortParameters = { m => m }
)(implicit p: Parameters) extends LazyModule {

  val node = TLAdapterNode(
    clientFn = c => c,
    managerFn = managerRewriteFn
  )

  lazy val module = new LazyModuleImp(this) {
    (node.in.map(_._1) zip node.out.map(_._1)).foreach { case (i, o) =>
      o.a <> i.a
      o.a.bits.address := rewriteFn(i.a.bits.address)

      i.b <> o.b
      i.b.bits.address := inverseFn(o.b.bits.address)

      o.c <> i.c
      o.c.bits.address := rewriteFn(i.c.bits.address)

      i.d <> o.d
      o.e <> i.e
    }
  }
}


object AddressOrNode {

  private def inverseOrAddressSet(
    a: AddressSet,
    baseAddr: BigInt
  ): Option[AddressSet] = {

    val conflict = baseAddr & ~a.mask & ~a.base

    if (conflict != 0) {
      None
    } else {
      Some(
        AddressSet(
          a.base & ~baseAddr,
          a.mask | baseAddr
        )
      )
    }
  }

  def apply(
    baseAddr: BigInt
  )(implicit
    p: Parameters,
    valName: ValName,
    sourceInfo: SourceInfo
  ): TLAdapterNode = {

    val rewriteManagers:
      TLSlavePortParameters => TLSlavePortParameters = { mp =>

      val managers = mp.managers.flatMap { m =>
        val localAddresses =
          m.address.flatMap(a => inverseOrAddressSet(a, baseAddr))

        if (localAddresses.isEmpty) {
          None
        } else {
          Some(m.v1copy(address = localAddresses))
        }
      }

      mp.v1copy(managers = managers)
    }

    LazyModule(
      new AddressRewriterNode(
        x => x | baseAddr.U,
        x => x & (~baseAddr.U).asUInt,
        rewriteManagers
      )
    ).node
  }
}


object AddressAndNode {

  def apply(
    baseAddr: BigInt
  )(implicit
    p: Parameters,
    valName: ValName,
    sourceInfo: SourceInfo
  ): TLAdapterNode = {

    LazyModule(
      new AddressRewriterNode(
        x => x & baseAddr.U,
        x => x | (~baseAddr.U).asUInt
      )
    ).node
  }
}
