# 📋 Milestone 3 - Next Steps Checklist

## Before Testing

- [ ] **Start Neo4j Database**
  - Open Neo4j Desktop
  - Start your database instance
  - Verify it's running at `neo4j://localhost:7687`

- [ ] **Verify config.txt**
  ```
  URI=neo4j://localhost:7687
  USERNAME=neo4j
  PASSWORD=your_password
  ```

- [ ] **Create Knowledge Graph**
  ```bash
  python create_kg.py
  ```
  Expected output: "Knowledge Graph successfully created."

## Testing Phase

### 1. Quick System Test
```bash
python quick_start_test.py
```
This will verify all components work correctly.

### 2. Test Individual Components (Optional)
```bash
# Component 1 - Input Preprocessing
python component_1_input_preprocessing.py

# Component 2 - Graph Retrieval  
python component_2_graph_retrieval.py

# Component 3 - LLM Layer (downloads ~500MB model on first run)
python component_3_llm_layer.py
```

### 3. Run the Full Application
```bash
streamlit run component_4_ui_app.py
```

Expected: Browser opens at `http://localhost:8501`

## In the Streamlit UI

- [ ] Click **"Load/Reload Models"** in sidebar
- [ ] Wait for "✅ Models ready" (2-5 minutes first time)
- [ ] Test with a predefined question
- [ ] Test with a custom question
- [ ] View all result sections (preprocessing, KG, embeddings, answer)
- [ ] Try changing models
- [ ] Export results as JSON

## Git Submission Preparation

### 1. Create Milestone3 Branch
```bash
git checkout -b Milestone3
```

### 2. Add All Files
```bash
git add .
git commit -m "Complete Milestone 3: All 4 components implemented"
```

### 3. Push to GitHub
```bash
git push origin Milestone3
```

### 4. Keep Repository Private (until deadline)
- Ensure repo is private until December 15, 23:59
- After deadline, make public or add `csen903w25-sys` as collaborator

## Submission Requirements ✅

### GitHub Repository
- [x] Branch named `Milestone3` created
- [x] All component files present
- [x] README.md with setup instructions
- [x] Requirements.txt with dependencies
- [x] Working code that can be run

### Components Checklist
- [x] Component 1: Input Preprocessing
- [x] Component 2: Graph Retrieval (baseline + embeddings)
- [x] Component 3: LLM Layer (3 models)
- [x] Component 4: Streamlit UI
- [x] 12+ Predefined Questions
- [x] Documentation of limitations

### Presentation Slides
- [ ] Create presentation covering:
  - System architecture
  - Each component explanation
  - Demo screenshots/video
  - Results and analysis
  - Limitations and future work
- [ ] Upload to Google Slides/PowerPoint
- [ ] Get shareable link

### Final Submission
- [ ] Submit GitHub repository link
- [ ] Submit presentation slides link
- [ ] Submit before December 15, 23:59

## Presentation Preparation (Dec 16+)

### What to Demo
1. **Architecture Overview** (2 min)
   - Show the 4 components and how they connect
   - Explain the airline use case

2. **Component 1: Input Preprocessing** (3 min)
   - Show intent classification working
   - Demonstrate entity extraction
   - Display embeddings

3. **Component 2: Graph Retrieval** (4 min)
   - Show baseline Cypher queries (pick 3-4 examples)
   - Demonstrate embedding-based retrieval
   - Compare the two approaches

4. **Component 3: LLM Layer** (4 min)
   - Show structured prompt format
   - Compare outputs from different models
   - Discuss quantitative metrics

5. **Component 4: UI Demo** (5 min)
   - Live demo of asking questions
   - Show different result sections
   - Demonstrate model switching

6. **Results & Analysis** (3 min)
   - What worked well
   - Challenges faced
   - Performance comparison

7. **Limitations & Future Work** (2 min)
   - Component limitations
   - Possible improvements

### Backup Plan
- Record a demo video in case of technical issues
- Have screenshots of key functionality
- Prepare offline version of slides

## Common Issues & Solutions

### Issue: "Module not found"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "Neo4j connection failed"
**Solution**:
1. Check Neo4j is running
2. Verify config.txt credentials
3. Test connection in Neo4j Browser

### Issue: "Out of memory"
**Solution**:
- Use smaller model: `google/flan-t5-base`
- Close other applications
- Enable swap/virtual memory

### Issue: "Model download too slow"
**Solution**:
- Use campus/home internet (not mobile hotspot)
- Models cached after first download
- Consider downloading overnight

### Issue: "Streamlit won't start"
**Solution**:
```bash
# Try different port
streamlit run component_4_ui_app.py --server.port 8502
```

## Performance Tips

### Speed up testing:
1. Use `google/flan-t5-base` (fastest model)
2. Disable embeddings for quick tests
3. Start with predefined questions
4. Cache is your friend - subsequent runs are faster

### For presentation:
1. Pre-load models before demo
2. Have example questions ready
3. Test on presentation computer beforehand
4. Have backup screenshots

## Final Verification

Before submission, verify:
- [ ] All Python files have no syntax errors
- [ ] README.md renders correctly on GitHub
- [ ] requirements.txt is complete
- [ ] Branch is named exactly "Milestone3"
- [ ] Repository is private (until deadline)
- [ ] All team member names in documentation

## Timeline Reminder

- **Dec 11**: You are here! ✅
- **Dec 12**: Testing and bug fixes
- **Dec 13**: Create presentation slides
- **Dec 14**: Practice demo
- **Dec 15, 23:59**: Submission deadline ⏰
- **Dec 16+**: Evaluation and presentation

## Support Resources

- **Office Hours**: Check CMS for schedule
- **Documentation**: README.md
- **Troubleshooting**: See README.md troubleshooting section
- **Course Account**: `csen903w25-sys` (add after deadline)

---

## Quick Commands Reference

```bash
# Setup
pip install -r requirements.txt
python create_kg.py

# Test
python quick_start_test.py

# Run individual components
python component_1_input_preprocessing.py
python component_2_graph_retrieval.py
python component_3_llm_layer.py

# Run full app
streamlit run component_4_ui_app.py

# Git operations
git checkout -b Milestone3
git add .
git commit -m "Complete Milestone 3"
git push origin Milestone3
```

---

**Good luck with your submission! 🎉**
