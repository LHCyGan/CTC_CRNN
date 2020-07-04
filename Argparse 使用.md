### Argparse Tutorial

这份代码中使用 Argparse库 来解析命令行参数  
官网链接：https://docs.python.org/3/howto/argparse.html#conclusion

#### 什么是Argparse库？  

the recommended command-line parsing module in the Python standard library.  
概要：  
Python标准库中的命令行解析模块。

#### Argparse的基本用法

```
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
```

parser = argparse.ArgumentParser() : 创建argparse对象，可用于设置参数  
parse_args()  : 其属性名是设置的各个参数名，属性值是传入参数的值。  
以上命令，自带--help参数，没有声明参数没有其他效果。  


#### Introducing Positional arguments

接收 命令行中的位置参数：  

```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",type=int)
args = parser.parse_args()
print(args.square**2)
```

help=  帮助信息  
type= 传入参数的数据类型  

使用add_argument()后，需要在命令行后面指定特定类型的值，否则会报错：  

```
$ python3 prog.py 4
16
$ python3 prog.py four
usage: prog.py [-h] square
prog.py: error: argument square: invalid int value: 'four'
```

#### Introducing Optional arguments

定义可选参数，使用命令行时可以传参也可以不传参。不传参时，其值为None。add_argument 定义可选参数时要加上--  

```
parser.add_argument("--verbosity", help="increase output verbosity")
```

使用命令行时，如果要给可选参数传参，语法是：  
--VariableName   Value

```
$ python3 prog.py --verbosity 1
verbosity turned on
$ python3 prog.py
$ python3 prog.py --help
usage: prog.py [-h] [--verbosity VERBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  --verbosity VERBOSITY
                        increase output verbosity
$ python3 prog.py --verbosity
usage: prog.py [-h] [--verbosity VERBOSITY]
prog.py: error: argument --verbosity: expected one argument
```

如果想对可选参数的值进行限制，使其只接受True 或者 False，可以使action="store_true"。当命令行传参的方式是：--verbose  并不指定其值时，为True；当不指定这个参数时，参数的值为False；当传入的是非布尔值时，报错。例：

```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")
```

Output:

```
$ python3 prog.py --verbose
verbosity turned on
$ python3 prog.py --verbose 1
usage: prog.py [-h] [--verbose]
prog.py: error: unrecognized arguments: 1
$ python3 prog.py --help
usage: prog.py [-h] [--verbose]

optional arguments:
  -h, --help  show this help message and exit
  --verbose   increase output verbosity
```

#### Short options

 设置可缩写的参数：  
 -ShortName , --LongName  
例：  

```
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
```

官网链接：  
https://docs.python.org/3/howto/argparse.html#short-options 


### Argparse module(更复杂的情况)

上面的Tutorial是命令行解析库Argparse入门的教程，如果遇到更复杂的情况，可以查这个库的API。  
https://docs.python.org/3/library/argparse.html#module-argparse    
https://docs.python.org/3/library/argparse.html#required  

以下是代码中出现的复杂命令行参数：  

#### required  

如果在add_argument()中加了required=True，那么这个参数不再是可选参数，而是必须要传值的参数。这种语法应该尽量要避免。  
https://docs.python.org/3/library/argparse.html#required  

#### default

如果命令行中没有出现某个参数，这个参数可以取defaule默认值。  
https://docs.python.org/3/library/argparse.html#default  

#### metavar

补充的帮助信息  
https://docs.python.org/3/library/argparse.html#metavar  
例：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', metavar='YYY')
>>> parser.add_argument('bar', metavar='XXX')
>>> parser.parse_args('X --foo Y'.split())
Namespace(bar='X', foo='Y')
>>> parser.print_help()
usage:  [-h] [--foo YYY] XXX

positional arguments:
 XXX

optional arguments:
 -h, --help  show this help message and exit
 --foo YYY
```

