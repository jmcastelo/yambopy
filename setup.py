from setuptools import setup

packages_yambopy = ['yambopy',
                    'yambopy.io',
                    'yambopy.dbs',
                    'yambopy.bse',
                    'yambopy.rt',
                    'yambopy.plot',
                    'yambopy.generic',
                    'qepy',
                    'schedulerpy',
                    'yamboparser']

if __name__ == '__main__':
    setup(name='yambopy',
          version='0.1',
          description='Automatic workflows for Yambo.',
          author='Henrique Miranda',
          author_email='miranda.henrique@gmail.com',
          requires=['numpy','matplotlib','netCDF4'],
          scripts=['scripts/yambopy'],
          packages=packages_yambopy,
          )
