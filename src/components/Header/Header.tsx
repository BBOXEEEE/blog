import React from "react";
import { Link } from "gatsby";
import * as styles from "./Header.module.scss";

type Props ={
    title: string;
};

const Header: React.FC<Props> = ({ title }) => {
    return (
        <header className={styles.header}>
            <Link to="/" className={styles.title}>{title}</Link>
            <div className={styles.divider} />
        </header>
    );
};

export default Header;
