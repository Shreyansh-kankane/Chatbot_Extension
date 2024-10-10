import { Router } from 'express';
import { s3Controller, createEmbeddings } from '../controllers/data.controller';  // Destructure both functions

const dataRouter = Router();

// Correct usage of the imported controller functions
dataRouter.post('/s3Data', s3Controller);
dataRouter.post('/createEmbeddings', createEmbeddings);

export default dataRouter;
