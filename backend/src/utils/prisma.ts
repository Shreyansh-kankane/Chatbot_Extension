import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
    // Write your database queries here.......
     
//   const newUser = await createUser();
//   console.log("newUser" , newUser);

//   const newBot = initializeBot() ;
//   console.log("new bot successfully created" , newBot.domain);
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })

export default prisma ;