// const db = import ('better-sqlite3')('database.db')
import Database from 'better-sqlite3';
const db = new Database('database.db');

const createTable = () => {
    const sql = `
    CREATE TABLE IF NOT EXISTS fruit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category CHARACTER(20),
        exec_time REAL    
    )
    `
    db.prepare(sql).run()
}

// createTable()

const insertTable = (category, exec_time) => {
    const sql = `
        INSERT INTO fruit (category, exec_time)
        VALUES (?, ?)
    `
    db.prepare(sql).run(category, exec_time)
}

// insertTable("Best", 9)

const getFruits = () => {
    const sql = `
    SELECT * FROM fruit
    `
    const rows = db.prepare(sql).all()
    console.log(rows)
}

getFruits()
