/*
  Warnings:

  - You are about to drop the column `role` on the `User` table. All the data in the column will be lost.
  - Added the required column `mobile` to the `User` table without a default value. This is not possible if the table is not empty.
  - Added the required column `password` to the `User` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "User" DROP COLUMN "role",
ADD COLUMN     "mobile" INTEGER NOT NULL,
ADD COLUMN     "password" TEXT NOT NULL,
ALTER COLUMN "namespace" DROP NOT NULL,
ALTER COLUMN "domain" DROP NOT NULL;

-- DropEnum
DROP TYPE "Role";
