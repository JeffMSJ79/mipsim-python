#!/usr/bin/env python
import sys
import struct
import time
import os

DEBUG = False
ENDIAN = "<"

NameList = {}
CallStack = []
logdir = str(time.time())
lastcount = 0

class ElfHeader:
	def __init__(self,binary):
		self.e_ident, self.e_type, self.e_machine, self.e_version, self.e_entry, self.e_phoff, self.e_shoff, self.e_flags, self.e_ehsize, self.e_phentsize, self.e_phnum, self.e_shentsize, self.e_shnum, self.e_shstrndx = struct.unpack("<16shhiiiiihhhhhh",binary)

class SectionHeader:
	def __init__(self,binary):
		self.sh_name, self.sh_type, self.sh_flags, self.sh_addr, self.sh_offset, self.sh_size, self.sh_link, self.sh_info, self.sh_addralign, self.sh_entsize = struct.unpack("<iiiiiiiiii",binary)

class SymbolTableEntry:
	def __init__(self,binary):
		self.st_name, self.st_value, self.st_size, self.st_info, self.st_other, self.st_shndx = struct.unpack("<IIIBBH",binary)

class NameEntry:
	def __init__(self,n_value,n_type,n_name):
		self.value = n_value
		self.type = n_type
		self.name = n_name

class Memory:
	def __init__(self):
		self.Mem = [[],[],[0]*2097152] # InstMem, Bss, Stack
		self.Ranges = [[0x400000,0xfffffff],[0x10000000,0x40000000],[0x80000000,0x80000000]]

	def FindRange(self,addr):
		index = 0
		while index<len(self.Ranges) and addr>self.Ranges[index][1]:
			index += 1
		return index
	
	def PutData(self,addr,data):
		index = self.FindRange(addr)
		datasize = len(data)/4
		offset = (addr-self.Ranges[index][0])/4
		diff = offset+datasize-len(self.Mem[index])
		if diff>0:
			self.Mem[index].extend([0]*(diff))
		if index!=1:
			self.Mem[index][offset:offset+datasize] = struct.unpack("<%dI" % datasize, data)

	def Read(self,addr,pc):
		index = self.FindRange(addr)
		offset = abs(addr-self.Ranges[index][0])/4
		assert offset<len(self.Mem[index]),"Memory Read out of boundary, index: %d, address: %x, range: %x, pc: %x" % (index,addr,len(self.Mem[index]),pc)
		return self.Mem[index][offset]

	def Write(self,addr,value,pc):
		index = self.FindRange(addr)
		offset = abs(addr-self.Ranges[index][0])/4
		assert offset<len(self.Mem[index]),"Memory Write out of boundary, index: %d, address: %x, range: %x, pc: %x" % (index,addr,len(self.Mem[index]),pc)
		self.Mem[index][offset] = value

class States:
	def __init__(self,pc,mem,resumeFile):
		self.regFile = [0]*32
		self.pc = pc
		self.mem = mem
		self.hi = 0
		self.lo = 0
		self.regFile[28] = 0x10008000
		self.regFile[29] = 0x80000000

		if resumeFile!="":
			f = open(resumeFile,"r")
			line = f.readline()
			regs = line.split("\t")
			for i in range(0,32):
				self.regFile[i] = int(regs[i])
			self.pc = int(regs[32])
			self.hi = int(regs[33])
			self.lo = int(regs[34])
			line = f.readline()
			sizes = line.split("\t")
			for i in range(0,3):
				size = int(sizes[i])
				self.mem.Mem[i] = [0]*size
				line = f.readline().split(" ")
				for j in range(0,size):
					self.mem.Mem[i][j] = int(line[j])
			f.close()

class Inst:
	def __init__(self,inst):
		self.opcode = inst>>26
		self.rs = (inst>>21) & 31
		self.rt = (inst>>16) & 31
		self.rd = (inst>>11) & 31
		self.amt = (inst>>6) & 31
		self.imm = inst & 0xffff
		self.optype = inst & 63
		self.target = inst & 0x3ffffff

def GetNameInStringTable(index,strtable):
	ret = ""
	length = len(strtable)
	while index<length and strtable[index]!='\0':
		ret += strtable[index]
		index += 1
	return ret

#Assume no arguments right now
def GetFileName():
	assert len(sys.argv)>=2, "Please provide the binary to simulate."
	fileName = sys.argv[1]
	return fileName

def GetElfHeader(f):
	eh = ElfHeader(f.read(52))
	assert eh.e_ident[0]!=0x7f or eh.e_ident[1]!='E' or eh.e_ident[2]!='l' or eh.e_ident[3]!='f',"Could not read ELF file header."
	return eh

def GetAllSections(f,eh):
	sections = []
	for i in range(0,eh.e_shnum):
		f.seek(eh.e_shoff+i*eh.e_shentsize)
		sh = SectionHeader(f.read(40))
		if sh.sh_type==0: #Unused Section
			continue
		sections.append(sh)
	return sections
		
def GetHeaderStringTable(f,sections):
	for i in range(0,len(sections)):
		sh = sections[i]	
		if sh.sh_type == 3: #Section String Table
			f.seek(sh.sh_offset)
			table, = struct.unpack("%ds" % (sh.sh_size,),f.read(sh.sh_size))
			if table[sh.sh_name:sh.sh_name+9]==".shstrtab":
				return table

def AnalyzeSymbolTable(symtab, stringtab):
	numSyms = len(symtab)/16
	for i in range(0,numSyms):
		entry = SymbolTableEntry(symtab[i*16:(i+1)*16])
		type = entry.st_info & 0xf
		bind = entry.st_info >> 4
		if (bind == 1 or bind == 2 or type == 1 or type == 2):
			nm = NameEntry(entry.st_value,type,GetNameInStringTable(entry.st_name,stringtab))
			NameList[nm.value] = nm.name
			if nm.name == "main":
				mainStart = nm.value
				mainEnd = nm.value + entry.st_size
	return (mainStart,mainEnd)

def AnalyzeSections(f,sections,shstrtab):
	mem = Memory()
	for i in range(0,len(sections)):
		sh = sections[i]
		name = GetNameInStringTable(sh.sh_name,shstrtab)
		f.seek(sh.sh_offset)
		data = f.read(sh.sh_size)
		if sh.sh_addr>=0x400000:
			if sh.sh_addr>=0x10000000:
				print name
			mem.PutData(sh.sh_addr,data)
		if name==".symtab":
			symtab = data
		elif name==".strtab":
			stringtab = data
	mainStart,mainEnd = AnalyzeSymbolTable(symtab, stringtab)
	return mem,mainStart,mainEnd
		
def ProcessBinary(f):
	eh = GetElfHeader(f)
	sections = GetAllSections(f,eh)
	shstrtab = GetHeaderStringTable(f,sections)
	mem,mainStart,mainEnd = AnalyzeSections(f,sections,shstrtab)
	print "Instruction Memory up to: %x" % (len(mem.Mem[0])*4+0x400000)
	print "Bss up to %x" % (len(mem.Mem[1])*4+0x10000000)
	return mem,mainStart,mainEnd

def Sign(num,bits):
	if num&(1<<(bits-1))>0:
		return num-(1<<bits)
	else:
		return num

def Unsign(num):
	if num>=0:
		return num
	else:
		return 0x100000000+num

def addiu(inst,states):	
	states.regFile[inst.rt] = Unsign((states.regFile[inst.rs]+Sign(inst.imm,16)) & 0xffffffff)

def sw(inst,states):	
	states.mem.Write(Sign(inst.imm,16)+states.regFile[inst.rs],states.regFile[inst.rt],states.pc)

def addu(inst,states):
	states.regFile[inst.rd] = Unsign((states.regFile[inst.rs]+states.regFile[inst.rt]) & 0xffffffff)

def subu(inst,states):
	states.regFile[inst.rd] = Unsign((states.regFile[inst.rs]-states.regFile[inst.rt]) & 0xffffffff)

def opand(inst,states):
	states.regFile[inst.rd] = states.regFile[inst.rs] & states.regFile[inst.rt]

def opor(inst,states):
	states.regFile[inst.rd] = states.regFile[inst.rs] | states.regFile[inst.rt]

def nor(inst,states):
	states.regFile[inst.rd] =~(states.regFile[inst.rs] | states.regFile[inst.rt])

def jr(inst,states):
	states.pc = states.regFile[inst.rs]
	return 1

def slt(inst,states):
	if Sign(states.regFile[inst.rs],32)<Sign(states.regFile[inst.rt],32):
		states.regFile[inst.rd] = 1
	else:
		states.regFile[inst.rd] = 0

def sltu(inst,states):
	if states.regFile[inst.rs]<states.regFile[inst.rt]:
		states.regFile[inst.rd] = 1
	else:
		states.regFile[inst.rd] = 0

def mult(inst,states):
	result = Sign(states.regFile[inst.rs],32)*Sign(states.regFile[inst.rt],32)
	states.lo = Unsign(result & 0xffffffff)
	states.hi = Unsign(result>>32)

def div(inst,states):
	op1 = Sign(states.regFile[inst.rs],32)
	op2 = Sign(states.regFile[inst.rt],32)
	states.lo = Unsign(op1/op2)
	states.hi = Unsign(op1%op2)

def multu(inst,states):
	result = states.regFile[inst.rs]*states.regFile[inst.rt]
	states.lo = Unsign(result & 0xffffffff)
	states.hi = Unsign(result>>32)

def mflo(inst,states):
	states.regFile[inst.rd] = states.lo

def mfhi(inst,states):
	states.regFile[inst.rd] = states.hi

def jalr(inst,states):
	states.regFile[31] = states.pc+8
	states.pc = states.regFile[inst.rs]
	return 1

def srl(inst,states):
	states.regFile[inst.rd] = states.regFile[inst.rt] >> inst.amt

def sra(inst,states):
	highbit = (states.regFile[inst.rt]&0x80000000)>>31
	dup = (highbit<<inst.amt)-highbit
	states.regFile[inst.rd] = (states.regFile[inst.rt] + (dup << 32)) >> inst.amt

def sll(inst,states):
	states.regFile[inst.rd] = (states.regFile[inst.rt] << inst.amt) & 0xffffffff

def sllv(inst,states):
	states.regFile[inst.rd] = (states.regFile[inst.rt] << states.regFile[inst.rs]) & 0xffffffff

def xor(inst,states):
	states.regFile[inst.rd] = states.regFile[inst.rt] ^ states.regFile[inst.rs]

#def syscall():
#	print "syscall seen, v0: %x at address %x" % (regFile[2],pc)

def arith(inst,states):
	localmap = {0:sll,9:jalr,33:addu,35:subu,8:jr,42:slt,24:mult,18:mflo, 43:sltu, 36:opand, 37:opor, 2:srl, 25:multu, 16:mfhi, 39:nor, 3:sra, 4:sllv, 38:xor, 26:div}
	assert inst.optype in localmap,"Unrecognized optype: %d, in address %x" % (inst.optype,states.pc)
	return localmap[inst.optype](inst,states)

def beq(inst,states):
	if states.regFile[inst.rs]==states.regFile[inst.rt]:
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1

def bne(inst,states):
	if states.regFile[inst.rs]!=states.regFile[inst.rt]:
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1

def lw(inst,states):
	states.regFile[inst.rt] = states.mem.Read(states.regFile[inst.rs]+Sign(inst.imm,16),states.pc)

def lb(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	base = addr&0xfffffffc
	offset = addr&3
	mask = 0xff<<(offset*8)
	states.regFile[inst.rt] = Sign((states.mem.Read(base,states.pc)&mask)>>(8*offset),8)

def sb(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	base = addr&0xfffffffc
	offset = addr&3
	mask = 0xff<<(offset*8)
	states.mem.Write(base,(states.mem.Read(base,states.pc)&(~mask))|((states.regFile[inst.rt]&0xff)<<(offset*8)),states.pc)

def sh(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	base = addr&0xfffffffc
	offset = addr&3
	mask = 0xffff<<(offset*8)
	states.mem.Write(base,(states.mem.Read(base,states.pc)&(~mask))|((states.regFile[inst.rt]&0xffff)<<(offset*8)),states.pc)

def lbu(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	base = addr&0xfffffffc
	offset = addr&3
	mask = 0xff<<(offset*8)
	states.regFile[inst.rt] = (states.mem.Read(base,states.pc)&mask)>>(8*offset)

def lhu(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	base = addr&0xfffffffc
	offset = addr&3
	mask = 0xffff<<(offset*8)
	states.regFile[inst.rt] = (states.mem.Read(base,states.pc)&mask)>>(8*offset)

def slti(inst,states):
	if Sign(states.regFile[inst.rs],32)<Sign(inst.imm,16):
		states.regFile[inst.rt] = 1
	else:
		states.regFile[inst.rt] = 0

def sltiu(inst,states):
	if states.regFile[inst.rs]<inst.imm:
		states.regFile[inst.rt] = 1
	else:
		states.regFile[inst.rt] = 0

def lui(inst,states):
	states.regFile[inst.rt] = (inst.imm<<16)&0xffffffff

def ori(inst,states):
	states.regFile[inst.rt] = states.regFile[inst.rs] | inst.imm

def andi(inst,states):
	states.regFile[inst.rt] = states.regFile[inst.rs] & inst.imm

def xori(inst,states):
	states.regFile[inst.rt] = states.regFile[inst.rs] ^ inst.imm

def jal(inst,states):
	states.regFile[31] = states.pc+8
	states.pc = ((states.pc&0xf0000000) | (inst.target<<2))
	return 1

def ble(inst,states):
	if Sign(states.regFile[inst.rs],32)<=Sign(states.regFile[inst.rt],32):
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1

def blt(inst,states):
	if Sign(states.regFile[inst.rs],32)<Sign(states.regFile[inst.rt],32):
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1

def bgt(inst,states):
	if Sign(states.regFile[inst.rs],32)>Sign(states.regFile[inst.rt],32):
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1

def beql(inst,states):
	if states.regFile[inst.rs]==states.regFile[inst.rt]:
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1
	else:
		return 2

def bnel(inst,states):
	if states.regFile[inst.rs]!=states.regFile[inst.rt]:
		states.pc = states.pc + 4 + Sign(inst.imm,16)*4
		return 1
	else:
		return 2

def j(inst,states):
	states.pc = (states.pc&0xf0000000) | (inst.target<<2)
	return 1

def swr(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	index = addr&3
	base = addr&0xfffffffc
	states.mem.Write(base,(states.mem.Read(base,states.pc)&(0xffffffff>>((4-index)*8)))|(states.regFile[inst.rt]&(0xffffffff<<(index*8))),states.pc)

def lwr(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	index = addr&3
	base = addr&0xfffffffc
	states.regFile[inst.rt] = (states.regFile[inst.rt] & (0xffffffff<<((4-index)*8))) | (states.mem.Read(base,states.pc) & (0xffffffff>>(index*8)))

def lwl(inst,states):
	addr = states.regFile[inst.rs]+Sign(inst.imm,16)
	index = addr&3
	base = addr&0xfffffffc
	states.regFile[inst.rt] = (states.regFile[inst.rt] & (0xffffffff>>((index+1)*8))) | (states.mem.Read(base,states.pc) & (0xffffffff<<((index+1)*8)))

def DumpRegisterFile(regFile):
	print "Register Dump:"
	for i in range(0,32):
		print "R",i," = ","%x" % regFile[i]

def log(states,count):
	global lastcount
	LOG = True
	#if states.pc in NameList:
		#print NameList[states.pc]
	if LOG and count-lastcount>=5000000:
		f = open(logdir+"/log","w")
		for i in range(0,32):
			f.write(str(states.regFile[i]))
			f.write("\t")
		f.write(str(states.pc)+"\t"+str(states.hi)+"\t"+str(states.lo)+"\n")
		f.write(str(len(states.mem.Mem[0]))+"\t"+str(len(states.mem.Mem[1]))+"\t"+str(len(states.mem.Mem[2]))+"\n")
		for i in range(0,3):
			for j in range(0,len(states.mem.Mem[i])):
				f.write(str(states.mem.Mem[i][j]))
				f.write(" ")
			f.write("\n")
		f.close()
		lastcount = count

def Simulate(states):	
	minsp = 0x80000000
	opmap = {0:arith, 1:blt, 2:j, 3:jal, 4:beq, 5:bne, 6:ble, 7:bgt, 9:addiu, 10:slti, 11:sltiu, 12:andi, 13:ori, 14: xori, 15:lui, 20: beql, 21: bnel, 32:lb, 34:lwl, 35:lw, 36:lbu, 37:lhu, 38:lwr, 40:sb, 41:sh, 43:sw, 46:swr}
	counter = 0
	jump = False
	while states.pc!=0:
		#print "%x" % pc
		#Fetch / Decode
		inst = Inst(mem.Read(states.pc,states.pc))

		#Execute
		#try:
		assert inst.opcode in opmap,"Unrecognized opcode: %d, in address %x" % (inst.opcode,states.pc)

		npc = states.pc + 4

		#0: Normal	1:Jump		2:Likely branch not taken
		status = opmap[inst.opcode](inst,states)

		#except Exception,err:
		#	print "Errors in address 0x%x" % pc
		#	raise err
		if jump:
			states.pc = temppc
			jump = False
			log(states,counter)
		elif status==2:
			states.pc = npc + 4
		elif status==1:
			jump = True
			temppc = states.pc
			states.pc = npc
		else:
			states.pc = npc

		counter += 1
		minsp = min(states.regFile[29],minsp)
	print "Number of instructions simulated: ",counter
	#if DEBUG:
	DumpRegisterFile(states.regFile)
	print "Minimum sp: 0x%x" % (minsp)
	print "Maximum Stack Size: %x" % (0x80000000-minsp)

def exportBinary(mem,start):
	names = ["instmem","bss","stack"]
	for i in range(0,3):
		f = open(names[i],"wb")
		for j in range(0,len(mem.Mem[i])):
			x = mem.Mem[i][j]
			f.write(chr(x&0xff))
			f.write(chr((x>>8)&0xff))
			f.write(chr((x>>16)&0xff))
			f.write(chr((x>>24)&0xff))
		f.close()
	print "PC starts at %x" % start

starttime = time.time()
fileName = GetFileName()
f = open(fileName,"rb")
mem,start,end = ProcessBinary(f)
f.close()
#exportBinary(mem,start)
if len(sys.argv)==3:
	resumeFile = sys.argv[2]
else:
	resumeFile = ""
states = States(start,mem,resumeFile)
os.mkdir(logdir)
Simulate(states)
endtime = time.time()
print "Time slapsed: ",str(endtime-starttime),"s"